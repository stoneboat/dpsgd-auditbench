#!/usr/bin/env python3
"""Generate in/out audit scores for white-box DP-FTRL auditing.

Trains a DP-FTRL model (centralized, tree-aggregated Gaussian noise) with
gradient-space dirac canaries on CIFAR-10 and saves both the SNR-optimal
"ancestor-sum" audit scores and the simpler "telescoping" final-model scores
at every checkpoint interval. Mirrors the layout of
`gen_scores_DP_whitebox.py` so downstream auditing scripts (e.g.
`run_auditing_comparison.py`) can pick up the same files.

Output per checkpoint epoch e (under exp_dir):
    in_scores_optimal_{e:06d}.csv     # SNR-optimal score, in canaries
    out_scores_optimal_{e:06d}.csv    # SNR-optimal score, out canaries
    in_scores_sum_{e:06d}.csv         # telescoping score, in canaries
    out_scores_sum_{e:06d}.csv        # telescoping score, out canaries
    in_scores_ndis_{e:06d}.csv        # SNR-optimal score / sqrt(L)
    out_scores_ndis_{e:06d}.csv       # SNR-optimal score / sqrt(L)
    privacy_params_{e:06d}.csv        # current_eps, delta
    inclusion_mask.csv                # bool[m] (canary in train set?)
    canary_leaves.csv                 # int[m]  (each canary's t_star)
    hparams.json
"""

import sys
import os
import math
import json
import secrets
import argparse
import logging
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))
src_dir = os.path.join(project_dir, "src")
sys.path.append(src_dir)

from utils import setup_logging, save_checkpoint, find_latest_checkpoint, load_checkpoint
from dataset import get_data_loaders
from network_arch import WideResNet
from train_dpftrl import train_dpftrl_whitebox, test
from whitebox_auditing.tree_mechanism import (
    tree_sigma_for_eps,
    tree_eps_for_sigma,
    num_levels,
)


# ==========================================
# Default Hyperparameters (parity with DP-SGD recipe)
# ==========================================
DEFAULT_LOGICAL_BATCH_SIZE = 4096
DEFAULT_MAX_PHYSICAL_BATCH_SIZE = 128
DEFAULT_AUG_MULTIPLICITY = 16
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_EPSILON = 8.0
DEFAULT_DELTA = 1e-5
DEFAULT_TARGET_STEPS = 2500
DEFAULT_EMA_DECAY = 0.9999
DEFAULT_LR = 4.0
DEFAULT_CKPT_INTERVAL = 20
DEFAULT_CANARY_COUNT = 5000
DEFAULT_PKEEP = 0.5


def main():
    parser = argparse.ArgumentParser(description="Generate scores for DP-FTRL auditing")
    parser.add_argument("--logical-batch-size", type=int, default=DEFAULT_LOGICAL_BATCH_SIZE)
    parser.add_argument("--max-physical-batch-size", type=int, default=DEFAULT_MAX_PHYSICAL_BATCH_SIZE)
    parser.add_argument("--aug-multiplicity", type=int, default=DEFAULT_AUG_MULTIPLICITY)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS,
                        help="Total DP-FTRL leaves; tree depth = ceil(log_2 target_steps).")
    parser.add_argument("--ema-decay", type=float, default=DEFAULT_EMA_DECAY)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--ckpt-interval", type=int, default=DEFAULT_CKPT_INTERVAL,
                        help="Score-snapshot interval, measured in DP-FTRL leaves.")
    parser.add_argument("--canary-count", type=int, default=DEFAULT_CANARY_COUNT)
    parser.add_argument("--pkeep", type=float, default=DEFAULT_PKEEP)
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

    exp_dir = os.path.join(
        args.data_dir,
        f"dpftrl-canaries-{DATABSEED}-{args.canary_count}-{args.pkeep}-cifar10",
    )
    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Experiment directory: {exp_dir}")
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Run experiment on device: {device}")

    torch_seed = int(DATABSEED % (2**32 - 1))
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    np.random.seed(torch_seed)
    rng = np.random.default_rng(torch_seed)
    logger.info(f"Set random seeds (torch, numpy) to: {torch_seed}")

    # Calibrate per-node sigma for the whole tree mechanism.
    sigma_node = tree_sigma_for_eps(args.epsilon, args.target_steps, args.delta)
    eps_check = tree_eps_for_sigma(sigma_node, args.target_steps, args.delta)
    logger.info(
        f"Tree mechanism: T={args.target_steps}, levels={num_levels(args.target_steps)}, "
        f"sigma_node={sigma_node:.4f}, target eps={args.epsilon}, eps roundtrip={eps_check:.4f}"
    )

    params = {
        "mechanism": "DP-FTRL (tree-aggregated Gaussian)",
        "logical_batch_size": args.logical_batch_size,
        "max_physical_batch_size": args.max_physical_batch_size,
        "aug_multiplicity": args.aug_multiplicity,
        "max_grad_norm": args.max_grad_norm,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "target_steps": args.target_steps,
        "tree_levels": num_levels(args.target_steps),
        "sigma_node": sigma_node,
        "ema_decay": args.ema_decay,
        "lr": args.lr,
        "ckpt_interval": args.ckpt_interval,
        "canary_count": args.canary_count,
        "pkeep": args.pkeep,
        "database_seed": DATABSEED,
    }
    with open(os.path.join(exp_dir, "hparams.json"), "w") as f:
        json.dump(params, f, indent=2)

    logger.info("Loading data...")
    # DP-FTRL has no Opacus BatchMemoryManager; build the loader with the
    # physical batch size so each forward pass fits in GPU memory. The
    # accumulation loop in train_dpftrl_whitebox composes physical batches
    # back up to logical_batch_size before each tree-mechanism step.
    train_loader, test_dataset = get_data_loaders(
        data_dir=args.data_dir,
        logical_batch_size=args.max_physical_batch_size,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024, shuffle=False, num_workers=args.num_workers
    )

    logger.info("Creating model...")
    model = WideResNet(depth=16, widen_factor=4).to(device)

    if args.ema_decay > 0:
        ema_model = WideResNet(depth=16, widen_factor=4).to(device).eval()
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.copy_(p.data)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        logger.info(f"EMA enabled with decay {args.ema_decay} (TIMM warmup)")
    else:
        ema_model = None
        logger.info("EMA disabled")

    # Build dirac canary directions: one (param_idx, flat_idx) per canary,
    # randomly distributed across model coords (no repeats).
    params_list = list(model.parameters())
    total_coords = sum(p.numel() for p in params_list)
    if args.canary_count > total_coords:
        raise ValueError(
            f"canary_count={args.canary_count} > total model coords {total_coords}"
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

    # Inclusion mask (Bernoulli(pkeep)) -- single coin flip per canary.
    mask_path = os.path.join(exp_dir, "inclusion_mask.csv")
    if os.path.isfile(mask_path):
        inclusion_mask = np.loadtxt(mask_path, delimiter=",").astype(bool)
        logger.info(f"Loaded existing inclusion mask from: {mask_path}")
    else:
        inclusion_mask = rng.random(args.canary_count) < args.pkeep
        np.savetxt(mask_path, inclusion_mask.astype(int), delimiter=",", fmt="%d")
        logger.info(f"Inclusion mask saved to: {mask_path}")

    # Canary leaf assignment: each canary fires at exactly one leaf.
    leaves_path = os.path.join(exp_dir, "canary_leaves.csv")
    if os.path.isfile(leaves_path):
        canary_leaves = np.loadtxt(leaves_path, delimiter=",").astype(np.int64)
        logger.info(f"Loaded existing leaf assignment from: {leaves_path}")
    else:
        canary_leaves = rng.integers(0, args.target_steps, size=args.canary_count, dtype=np.int64)
        np.savetxt(leaves_path, canary_leaves, delimiter=",", fmt="%d")
        logger.info(f"Canary leaf assignment saved to: {leaves_path}")

    n_in = int(inclusion_mask.sum())
    n_out = args.canary_count - n_in
    logger.info(f"Canaries: {n_in} IN, {n_out} OUT (pkeep={args.pkeep})")

    canary_scale = float(args.max_grad_norm)

    logger.info("Starting DP-FTRL training...")
    logger.info(f"  T (leaves): {args.target_steps}")
    logger.info(f"  LR: {args.lr}")
    logger.info(f"  Logical batch size: {args.logical_batch_size}")
    logger.info(f"  Max physical batch size: {args.max_physical_batch_size}")
    logger.info(f"  Aug multiplicity: {args.aug_multiplicity}")
    logger.info(f"  sigma_node: {sigma_node:.4f}")
    logger.info(f"  Canary count: {args.canary_count}, P(keep): {args.pkeep}")

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
        sigma_node=sigma_node,
        canary_dirac_indices=canary_dirac_indices,
        canary_leaf_assignment=canary_leaves,
        canary_inclusion_mask=inclusion_mask,
        canary_scale=canary_scale,
        ema_model=ema_model,
        ema_decay=args.ema_decay,
        ema_step_offset=0,
        logger=logger,
    )

    logger.info(f"Training complete: {leaves_done} leaves processed")

    # ---- final scores ----
    optimal = state.compute_optimal_scores(leaves_done)
    telescope = state.telescope_scores()
    in_optimal = optimal[inclusion_mask]
    out_optimal = optimal[~inclusion_mask]
    in_telescope = telescope[inclusion_mask]
    out_telescope = telescope[~inclusion_mask]

    L = num_levels(args.target_steps) + 1   # # ancestors of any leaf in the canonical tree
    # Normalized score for NDIS (the closed form expects unit-variance scaling
    # at the per-observation level; sqrt(L) standardizes the optimal score).
    in_ndis = optimal[inclusion_mask] / math.sqrt(L)
    out_ndis = optimal[~inclusion_mask] / math.sqrt(L)

    e = leaves_done
    np.savetxt(os.path.join(exp_dir, f"in_scores_optimal_{e:06d}.csv"), in_optimal, delimiter=",")
    np.savetxt(os.path.join(exp_dir, f"out_scores_optimal_{e:06d}.csv"), out_optimal, delimiter=",")
    np.savetxt(os.path.join(exp_dir, f"in_scores_sum_{e:06d}.csv"), in_telescope, delimiter=",")
    np.savetxt(os.path.join(exp_dir, f"out_scores_sum_{e:06d}.csv"), out_telescope, delimiter=",")
    np.savetxt(os.path.join(exp_dir, f"in_scores_ndis_{e:06d}.csv"), in_ndis, delimiter=",")
    np.savetxt(os.path.join(exp_dir, f"out_scores_ndis_{e:06d}.csv"), out_ndis, delimiter=",")
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

    logger.info(
        f"Saved scores: optimal in({n_in}) out({n_out}); telescope in({n_in}) out({n_out})."
    )
    logger.info(f"Final log file saved at: {log_file}")


if __name__ == "__main__":
    main()
