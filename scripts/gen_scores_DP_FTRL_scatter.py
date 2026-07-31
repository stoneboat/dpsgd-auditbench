#!/usr/bin/env python3
import sys
import os
import math
import json
import secrets
import argparse
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))
src_dir = os.path.join(project_dir, "src")
sys.path.append(src_dir)


from functools import partial

from utils import setup_logging
from dataset import get_data_loaders
from scatter_network import ScatterLinear
from train_dpftrl import train_dpftrl_whitebox, test, DPFTRLState, MFDPFTRLState
from whitebox_auditing.tree_mechanism import (tree_sigma_for_eps, tree_eps_for_sigma, num_levels)
from whitebox_auditing import matrix_factorization as mf


# ==========================================
# Defaults: single-pass DP-FTRL on CIFAR-10 with ScatterLinear.
#   N=50000, B=500  -> T = 100 leaves (one full pass).
# ==========================================
DEFAULT_LOGICAL_BATCH_SIZE = 500
DEFAULT_MAX_PHYSICAL_BATCH_SIZE = 500
DEFAULT_AUG_MULTIPLICITY = 1
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_EPSILON = 8.0
DEFAULT_DELTA = 1e-5
DEFAULT_TARGET_STEPS = 100
DEFAULT_EMA_DECAY = 0.0
DEFAULT_LR = 1.0
DEFAULT_CANARY_COUNT = 5000
DEFAULT_PKEEP = 0.5
DEFAULT_J = 2


def main():
    parser = argparse.ArgumentParser(
        description="Generate scores for DP-FTRL auditing on ScatterLinear (single-pass).",
    )
    parser.add_argument("--logical-batch-size", type=int, default=DEFAULT_LOGICAL_BATCH_SIZE)
    parser.add_argument("--max-physical-batch-size", type=int, default=DEFAULT_MAX_PHYSICAL_BATCH_SIZE)
    parser.add_argument("--aug-multiplicity", type=int, default=DEFAULT_AUG_MULTIPLICITY)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS,
                        help="Total DP-FTRL leaves; tree depth = ceil(log_2 target_steps).")
    parser.add_argument("--ema-decay", type=float, default=DEFAULT_EMA_DECAY,
                        help="EMA decay; <=0 disables EMA (default for single-pass).")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--canary-count", type=int, default=DEFAULT_CANARY_COUNT)
    parser.add_argument("--pkeep", type=float, default=DEFAULT_PKEEP)
    parser.add_argument("--scattering-J", type=int, default=DEFAULT_J,
                        help="Number of scales for the kymatio scattering transform.")
    parser.add_argument("--mechanism", choices=["tree", "mf"], default="tree",
                        help="tree = Honaker binary tree; mf = matrix factorization (banded sqrt).")
    parser.add_argument("--mf-bands", type=int, default=0,
                        help="Band the sqrt strategy at this lag; 0 = full square root.")
    parser.add_argument("--random-leaves", action="store_true",
                        help="Randomize canary leaves. Under mf the column norm of C varies with "
                             "the leaf, so this makes the pooled in-population a mixture; the "
                             "default puts every canary at leaf 0 to keep it exactly Gaussian.")
    parser.add_argument("--database-seed", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger, log_file = setup_logging(log_file=args.log_file, log_dir=args.log_dir)

    if args.database_seed is None:
        DATABSEED = secrets.randbits(128)
        logger.info(f"Generated random 128-bit seed: {DATABSEED}")
    else:
        DATABSEED = int(args.database_seed)
        logger.info(f"Using provided database seed: {DATABSEED}")

    tag = "" if args.mechanism == "tree" else f"-{args.mechanism}"
    exp_dir = os.path.join(
        args.data_dir,
        f"dpftrl-scatter-canaries-{DATABSEED}-{args.canary_count}-{args.pkeep}-cifar10{tag}",
    )
    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Experiment directory: {exp_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Run experiment on device: {device}")

    torch_seed = int(DATABSEED % (2**32 - 1))
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    np.random.seed(torch_seed)
    rng = np.random.default_rng(torch_seed)
    logger.info(f"Set random seeds (torch, numpy) to: {torch_seed}")

    strategy = decoder_mat = None
    if args.mechanism == "mf":
        strategy = mf.sqrt_strategy(args.target_steps, bands=args.mf_bands)
        decoder_mat = mf.decoder(strategy)
        sens = mf.sensitivity(strategy, clip_norm=args.max_grad_norm)
        sigma = mf.sigma_for_eps(args.epsilon, sens, args.delta)
        eps_check = mf.eps_for_sigma(sigma, sens, args.delta)
        tree_sigma = tree_sigma_for_eps(args.epsilon, args.target_steps, args.delta)
        logger.info(f"MF mechanism: T={args.target_steps}, bands={args.mf_bands or 'full'}, sens={sens:.4f}, sigma={sigma:.4f}, eps roundtrip={eps_check:.4f}")
        logger.info(f"  tree sensitivity={math.sqrt(num_levels(args.target_steps) + 1):.4f} sigma={tree_sigma:.4f} -> MF needs {tree_sigma / sigma:.2f}x less noise")
        sigma_node = sigma
    else:
        sigma_node = tree_sigma_for_eps(args.epsilon, args.target_steps, args.delta)
        eps_check = tree_eps_for_sigma(sigma_node, args.target_steps, args.delta)
        logger.info(f"Tree mechanism: T={args.target_steps}, levels={num_levels(args.target_steps)}, sigma_node={sigma_node:.4f}, target eps={args.epsilon}, eps roundtrip={eps_check:.4f}")

    params_summary = {
        "mechanism": args.mechanism,
        "mf_bands": args.mf_bands,
        "random_leaves": args.random_leaves,
        "model": f"ScatterLinear (kymatio J={args.scattering_J})",
        "logical_batch_size": args.logical_batch_size,
        "max_physical_batch_size": args.max_physical_batch_size,
        "aug_multiplicity": args.aug_multiplicity,
        "max_grad_norm": args.max_grad_norm,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "target_steps": args.target_steps,
        "tree_levels": num_levels(args.target_steps),
        "sigma_node": sigma_node,
        "mf_sensitivity": None if args.mechanism == "tree" else mf.sensitivity(strategy, args.max_grad_norm),
        "ema_decay": args.ema_decay,
        "lr": args.lr,
        "canary_count": args.canary_count,
        "pkeep": args.pkeep,
        "scattering_J": args.scattering_J,
        "database_seed": DATABSEED,
    }
    with open(os.path.join(exp_dir, "hparams.json"), "w") as f:
        json.dump(params_summary, f, indent=2)

    logger.info("Loading data...")
    train_loader, test_dataset = get_data_loaders(
        data_dir=args.data_dir,
        logical_batch_size=args.max_physical_batch_size,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024, shuffle=False, num_workers=args.num_workers
    )

    logger.info("Creating ScatterLinear model...")
    model = ScatterLinear(num_classes=10, image_size=32, J=args.scattering_J).to(device)
    logger.info(f"  feat_dim = {model.feat_dim}")

    if args.ema_decay > 0:
        ema_model = ScatterLinear(num_classes=10, image_size=32, J=args.scattering_J).to(device).eval()
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.copy_(p.data)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        logger.info(f"EMA enabled with decay {args.ema_decay}")
    else:
        ema_model = None
        logger.info("EMA disabled (single-pass default)")

    params_list = [p for p in model.parameters() if p.requires_grad]
    total_coords = sum(p.numel() for p in params_list)
    if args.canary_count > total_coords:
        raise ValueError(
            f"canary_count={args.canary_count} > learnable coords {total_coords}"
        )
    flat_offsets = np.cumsum([0] + [p.numel() for p in params_list])
    chosen_flat = rng.choice(total_coords, size=args.canary_count, replace=False)
    chosen_flat.sort()
    canary_dirac_indices = []
    for f in chosen_flat:
        p_idx = int(np.searchsorted(flat_offsets[1:], f, side="right"))
        within = int(f - flat_offsets[p_idx])
        canary_dirac_indices.append((p_idx, within))
    np.savetxt(os.path.join(exp_dir, "canary_coords.csv"),
               np.array(canary_dirac_indices, dtype=int), delimiter=",", fmt="%d",
               header="param_idx,flat_idx", comments="")

    mask_path = os.path.join(exp_dir, "inclusion_mask.csv")
    if os.path.isfile(mask_path):
        inclusion_mask = np.loadtxt(mask_path, delimiter=",").astype(bool)
        logger.info(f"Loaded existing inclusion mask from: {mask_path}")
    else:
        inclusion_mask = rng.random(args.canary_count) < args.pkeep
        np.savetxt(mask_path, inclusion_mask.astype(int), delimiter=",", fmt="%d")
        logger.info(f"Inclusion mask saved to: {mask_path}")

    leaves_path = os.path.join(exp_dir, "canary_leaves.csv")
    if os.path.isfile(leaves_path):
        canary_leaves = np.loadtxt(leaves_path, delimiter=",").astype(np.int64)
        logger.info(f"Loaded existing leaf assignment from: {leaves_path}")
    elif args.random_leaves or args.mechanism == "tree":
        # The tree gives every leaf the same ancestor-path length, so random leaves
        # stay identically distributed. MF has no such symmetry -- see --random-leaves.
        canary_leaves = rng.integers(0, args.target_steps, size=args.canary_count, dtype=np.int64)
        np.savetxt(leaves_path, canary_leaves, delimiter=",", fmt="%d")
        logger.info(f"Canary leaf assignment saved to: {leaves_path}")
    else:
        canary_leaves = np.zeros(args.canary_count, dtype=np.int64)
        np.savetxt(leaves_path, canary_leaves, delimiter=",", fmt="%d")
        logger.info("All canaries at leaf 0 (equal ||C[:, t]||, so the pooled population is homogeneous)")

    n_in = int(inclusion_mask.sum())
    n_out = args.canary_count - n_in
    logger.info(f"Canaries: {n_in} IN, {n_out} OUT (pkeep={args.pkeep})")

    canary_scale = float(args.max_grad_norm)

    logger.info("Starting DP-FTRL training (ScatterLinear, single-pass)...")
    logger.info(f"  T (leaves): {args.target_steps}")
    logger.info(f"  LR: {args.lr}")
    logger.info(f"  Logical batch size: {args.logical_batch_size}")
    logger.info(f"  Max physical batch size: {args.max_physical_batch_size}")
    logger.info(f"  Aug multiplicity: {args.aug_multiplicity}")
    logger.info(f"  sigma_node: {sigma_node:.4f}")
    logger.info(f"  Canary count: {args.canary_count}, P(keep): {args.pkeep}")

    canary_kw = dict(target_steps=args.target_steps, canary_dirac_indices=canary_dirac_indices,
                     canary_leaf_assignment=canary_leaves, canary_inclusion_mask=inclusion_mask,
                     canary_scale=canary_scale)
    if args.mechanism == "mf":
        state_builder = partial(MFDPFTRLState, sigma=sigma_node, strategy=strategy, decoder=decoder_mat, **canary_kw)
    else:
        state_builder = partial(DPFTRLState, sigma_node=sigma_node, **canary_kw)

    losses, state, leaves_done = train_dpftrl_whitebox(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        aug_multiplicity=args.aug_multiplicity,
        max_physical_batch_size=args.max_physical_batch_size,
        logical_batch_size=args.logical_batch_size,
        target_steps=args.target_steps,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        state_builder=state_builder,
        ema_model=ema_model,
        ema_decay=args.ema_decay,
        ema_step_offset=0,
        logger=logger,
    )

    logger.info(f"Training complete: {leaves_done} leaves processed")

    # ---- final scores ----
    optimal = state.compute_optimal_scores(leaves_done)
    in_optimal = optimal[inclusion_mask]
    out_optimal = optimal[~inclusion_mask]

    # MF scores are already divided by ||C[:, t]||, so their sd is sigma directly;
    # the tree score sums L ancestors and needs sqrt(L).
    ndis_scale = 1.0 if args.mechanism == "mf" else math.sqrt(num_levels(args.target_steps) + 1)
    in_ndis = in_optimal / ndis_scale
    out_ndis = out_optimal / ndis_scale

    if args.mechanism == "mf" and args.canary_count:
        mu0, mu1, sd = mf.audit_gaussian_params(strategy, sigma_node, int(canary_leaves[0]), canary_scale)
        logger.info(f"MF analytic pair (leaf {canary_leaves[0]}): out~N({mu0:.3f}, {sd:.3f}^2), in~N({mu1:.3f}, {sd:.3f}^2)")
        logger.info(f"MF empirical: out mean={out_ndis.mean():.3f} sd={out_ndis.std():.3f}, in mean={in_ndis.mean():.3f} sd={in_ndis.std():.3f}")

    # Andrew et al. (2024, Alg 3): max-over-leaves cosine of dirac canary
    # direction with the released noisy_G_t. One scalar per canary.
    andrew = state.compute_andrew_scores(leaves_done)
    in_andrew = andrew[inclusion_mask]
    out_andrew = andrew[~inclusion_mask]

    e = leaves_done
    np.savetxt(os.path.join(exp_dir, f"in_scores_optimal_{e:06d}.csv"), in_optimal, delimiter=",")
    np.savetxt(os.path.join(exp_dir, f"out_scores_optimal_{e:06d}.csv"), out_optimal, delimiter=",")
    np.savetxt(os.path.join(exp_dir, f"in_scores_ndis_{e:06d}.csv"), in_ndis, delimiter=",")
    np.savetxt(os.path.join(exp_dir, f"out_scores_ndis_{e:06d}.csv"), out_ndis, delimiter=",")
    np.savetxt(os.path.join(exp_dir, f"in_scores_andrew_{e:06d}.csv"), in_andrew, delimiter=",")
    np.savetxt(os.path.join(exp_dir, f"out_scores_andrew_{e:06d}.csv"), out_andrew, delimiter=",")
    np.savetxt(
        os.path.join(exp_dir, f"privacy_params_{e:06d}.csv"),
        [[args.epsilon, args.delta]],
        delimiter=",", header="current_eps,delta", comments="",
    )

    if ema_model is not None:
        live_acc = test(model, test_loader, device)
        ema_acc = test(ema_model, test_loader, device)
        logger.info(f"Final test acc: live={live_acc:.2f}%, EMA={ema_acc:.2f}%")
    else:
        acc = test(model, test_loader, device)
        logger.info(f"Final test acc: {acc:.2f}%")

    logger.info(f"Saved scores: optimal/ndis/andrew in({n_in}) out({n_out}).")
    logger.info(f"Final log file saved at: {log_file}")


if __name__ == "__main__":
    main()