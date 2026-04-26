#!/usr/bin/env python3
"""
Generate in/out audit scores for white-box DP-SGD auditing.
Trains a DP model with dirac canaries and saves in_scores, out_scores, and
privacy_params at each checkpoint interval (same logic as train_auditable_DP_model_whitebox.ipynb).
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
from opacus import PrivacyEngine
import torch.optim as optim

# Add the src directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)

from utils import setup_logging, save_checkpoint, find_latest_checkpoint, load_checkpoint
from dataset import get_data_loaders
from network_arch import WideResNet
from train import test, train_whitebox

# ==========================================
# Default Hyperparameters (white-box, from notebook)
# ==========================================
DEFAULT_LOGICAL_BATCH_SIZE = 4096
DEFAULT_MAX_PHYSICAL_BATCH_SIZE = 128
DEFAULT_AUG_MULTIPLICITY = 1
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_EPSILON = 8.0
DEFAULT_DELTA = 1e-5
DEFAULT_EPOCHS = 200
DEFAULT_TARGET_STEPS = 2500   # Sander/Mahloujifar setting
DEFAULT_EMA_DECAY = 0.9999    # Sander/Mahloujifar setting
DEFAULT_LR = 4.0
DEFAULT_MOMENTUM = 0.0
DEFAULT_CKPT_INTERVAL = 20
DEFAULT_CANARY_COUNT = 10000
DEFAULT_PKEEP = 0.5


def main():
    parser = argparse.ArgumentParser(
        description='Generate in/out audit scores for white-box DP-SGD (dirac canaries)'
    )
    # Hyperparameters
    parser.add_argument('--logical-batch-size', type=int, default=DEFAULT_LOGICAL_BATCH_SIZE)
    parser.add_argument('--max-physical-batch-size', type=int, default=DEFAULT_MAX_PHYSICAL_BATCH_SIZE)
    parser.add_argument('--aug-multiplicity', type=int, default=DEFAULT_AUG_MULTIPLICITY)
    parser.add_argument('--max-grad-norm', type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument('--epsilon', type=float, default=DEFAULT_EPSILON)
    parser.add_argument('--delta', type=float, default=DEFAULT_DELTA)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Hard cap on epochs; training also stops when --target-steps is reached.')
    parser.add_argument('--target-steps', type=int, default=DEFAULT_TARGET_STEPS,
                        help='Total DP-SGD steps to run (Sander/Mahloujifar use 2500). Privacy budget is calibrated to this exact step count.')
    parser.add_argument('--ema-decay', type=float, default=DEFAULT_EMA_DECAY,
                        help='Decay for the exponential moving average of model weights used at evaluation. Set to 0 to disable.')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--momentum', type=float, default=DEFAULT_MOMENTUM)
    parser.add_argument('--ckpt-interval', type=int, default=DEFAULT_CKPT_INTERVAL)
    # Experiment
    parser.add_argument('--canary-count', type=int, default=DEFAULT_CANARY_COUNT)
    parser.add_argument('--pkeep', type=float, default=DEFAULT_PKEEP)
    parser.add_argument('--database-seed', type=str, default=None,
                        help='128-bit integer as string; if None, generate random')
    # Paths
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--log-file', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger, log_file = setup_logging(log_file=args.log_file, log_dir=args.log_dir)

    if args.database_seed is None:
        DATABSEED = secrets.randbits(128)
        logger.info(f"Generated random 128-bit seed: {DATABSEED}")
    else:
        try:
            DATABSEED = int(args.database_seed)
        except ValueError:
            logger.error(f"Invalid database-seed: '{args.database_seed}'. Must be a valid integer.")
            sys.exit(1)
        logger.info(f"Using provided database seed: {DATABSEED}")

    exp_dir = os.path.join(
        args.data_dir,
        f"mislabeled-canaries-{DATABSEED}-{args.canary_count}-{args.pkeep}-cifar10",
    )
    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Experiment directory: {exp_dir}")

    ckpt_dir = os.path.join(exp_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Run experiment on device: {device}")

    torch_seed = int(DATABSEED % (2**32 - 1))
    np_seed = int(DATABSEED % (2**32 - 1))
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
    np.random.seed(np_seed)
    rng = np.random.default_rng(np_seed)
    logger.info(f"Set random seeds (torch, numpy) to: {torch_seed} (from DATABSEED: {DATABSEED})")

    params = {
        'logical_batch_size': args.logical_batch_size,
        'max_physical_batch_size': args.max_physical_batch_size,
        'aug_multiplicity': args.aug_multiplicity,
        'max_grad_norm': args.max_grad_norm,
        'epsilon': args.epsilon,
        'delta': args.delta,
        'epochs': args.epochs,
        'target_steps': args.target_steps,
        'ema_decay': args.ema_decay,
        'lr': args.lr,
        'momentum': args.momentum,
        'ckpt_interval': args.ckpt_interval,
        'canary_count': args.canary_count,
        'pkeep': args.pkeep,
        'database_seed': DATABSEED,
    }
    hparams_path = os.path.join(exp_dir, 'hparams.json')
    with open(hparams_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Hyperparameters saved to: {hparams_path}")

    logger.info("Loading data...")
    train_loader, test_dataset = get_data_loaders(
        data_dir=args.data_dir,
        logical_batch_size=args.logical_batch_size,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024, shuffle=False, num_workers=args.num_workers
    )

    logger.info("Creating model...")
    model = WideResNet(depth=16, widen_factor=4).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Calibrate the privacy budget to the *target step count*, not to args.epochs.
    # Opacus computes sigma from (target_eps, target_delta, sample_rate, epochs);
    # total budgeted steps = epochs * len(orig_loader). We pick epochs so the
    # budget covers exactly args.target_steps.
    steps_per_epoch_pre = max(1, len(train_loader))
    epochs_for_priv = max(
        1, math.ceil(args.target_steps / steps_per_epoch_pre)
    )
    epochs_cap = min(args.epochs, epochs_for_priv)
    logger.info(
        f"steps_per_epoch (pre-Opacus) = {steps_per_epoch_pre}, "
        f"target_steps = {args.target_steps}, "
        f"epochs_for_privacy = {epochs_for_priv}, epochs_cap = {epochs_cap}"
    )

    logger.info("Setting up privacy engine...")
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs_for_priv,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        max_grad_norm=args.max_grad_norm,
    )
    noise_multiplier = optimizer.noise_multiplier
    logger.info(f"Opacus computed noise_multiplier={noise_multiplier:.4f} for epsilon={args.epsilon}, delta={args.delta}, epochs={epochs_for_priv}")

    # ---- EMA: Sander et al. / Mahloujifar et al. eval on EMA weights, not live weights.
    if args.ema_decay > 0:
        ema_model = WideResNet(depth=16, widen_factor=4).to(device).eval()
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.copy_(p.data)
            for b_ema, b in zip(ema_model.buffers(), model.buffers()):
                b_ema.data.copy_(b.data)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        logger.info(f"EMA enabled with decay {args.ema_decay}")
    else:
        ema_model = None
        logger.info("EMA disabled (--ema-decay 0)")
    params['noise_multiplier'] = noise_multiplier
    with open(hparams_path, 'w') as f:
        json.dump(params, f, indent=2)

    # Checkpoint loading
    start_epoch = 1
    total_steps = 0
    sum_scores = None

    checkpoint_result = find_latest_checkpoint(ckpt_dir)
    if checkpoint_result is not None:
        checkpoint_path, checkpoint_epoch = checkpoint_result
        logger.info(f"Loading checkpoint '{checkpoint_path}' (epoch {checkpoint_epoch})...")
        loaded_epoch, loaded_global_step = load_checkpoint(
            checkpoint_path, model, optimizer, device, logger
        )
        start_epoch = loaded_epoch + 1
        steps_per_epoch = len(train_loader)
        sample_rate = 1 / len(train_loader)
        privacy_engine.accountant.history.append(
            (noise_multiplier, sample_rate, loaded_global_step)
        )
        logger.info(f"Resumed from Epoch {start_epoch}")
        logger.info(f"Privacy Accountant updated with {loaded_global_step} past steps.")
        total_steps = loaded_global_step if loaded_global_step is not None else 0

        # Restore EMA state if present
        if ema_model is not None:
            ema_path = os.path.join(ckpt_dir, f'ema_state_{checkpoint_epoch:06d}.pt')
            if os.path.isfile(ema_path):
                ema_model.load_state_dict(torch.load(ema_path, map_location=device))
                logger.info(f"Loaded EMA state from {ema_path}")
            else:
                logger.info(
                    f"No EMA state at {ema_path}; re-initializing EMA from current model "
                    "(estimate will be biased for the first few hundred steps)."
                )

        # Restore sum_scores from saved per-canary sums + inclusion mask
        sum_scores_path = os.path.join(exp_dir, f'sum_scores_{checkpoint_epoch:06d}.csv')
        if os.path.isfile(sum_scores_path) and total_steps > 0:
            sum_scores = np.loadtxt(sum_scores_path, delimiter=',')
            logger.info(
                f"Loaded sum_scores for epoch {checkpoint_epoch} (total_steps={total_steps})"
            )

        current_eps = privacy_engine.get_epsilon(args.delta)
        logger.info(f"Current Cumulative Epsilon: {current_eps:.2f}")
    else:
        logger.info("No checkpoint found. Starting from scratch.")

    # Build dirac canaries (same as notebook)
    params_list = list(model.parameters())
    canary_dirac_indices = []
    remaining = args.canary_count
    for p_idx, p in enumerate(params_list):
        take = min(remaining, p.numel())
        canary_dirac_indices.extend((p_idx, i) for i in range(take))
        remaining -= take
        if remaining == 0:
            break

    # Generate or load inclusion mask: True = IN (signal injected), False = OUT (pure noise)
    mask_path = os.path.join(exp_dir, 'inclusion_mask.csv')
    if os.path.isfile(mask_path):
        inclusion_mask = np.loadtxt(mask_path, delimiter=',').astype(bool)
        logger.info(f"Loaded existing inclusion mask from: {mask_path}")
    else:
        inclusion_mask = rng.random(args.canary_count) < args.pkeep
        np.savetxt(mask_path, inclusion_mask.astype(int), delimiter=',', fmt='%d')
        logger.info(f"Inclusion mask saved to: {mask_path}")
    n_in = int(inclusion_mask.sum())
    n_out = args.canary_count - n_in
    logger.info(f"Inclusion mask: {n_in} IN canaries, {n_out} OUT canaries (pkeep={args.pkeep})")

    logger.info("Starting training...")
    logger.info(f"  Epochs: {args.epochs}, LR: {args.lr}, Momentum: {args.momentum}")
    logger.info(f"  Logical batch size: {args.logical_batch_size}")
    logger.info(f"  Max physical batch size: {args.max_physical_batch_size}")
    logger.info(f"  Aug multiplicity: {args.aug_multiplicity}")
    logger.info(f"  Noise multiplier (computed): {noise_multiplier:.4f}")
    logger.info(f"  Canary count: {args.canary_count}, P(keep): {args.pkeep}")
    logger.info(f"  Checkpoint interval: {args.ckpt_interval} epochs")

    final_test_acc = None
    canary_prob = 1.0 / len(train_loader)

    for epoch in range(start_epoch, epochs_cap + 1):
        steps_remaining = args.target_steps - total_steps
        if steps_remaining <= 0:
            logger.info(f"Reached target_steps={args.target_steps} at epoch {epoch-1}; stopping.")
            break
        train_loss, num_steps, scores = train_whitebox(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            args.aug_multiplicity,
            args.max_physical_batch_size,
            args.logical_batch_size,
            canary_dirac_indices=canary_dirac_indices,
            canary_inclusion_mask=inclusion_mask,
            canary_prob=canary_prob,
            return_scores=True,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
            max_logical_steps=steps_remaining,
        )
        scores = np.asarray(scores)  # (num_steps, num_canaries)
        assert scores.shape[0] == num_steps, (
            f"scores.shape[0] = {scores.shape[0]} != num_steps = {num_steps}"
        )
        if sum_scores is None:
            sum_scores = scores.sum(axis=0)
        else:
            sum_scores = sum_scores + scores.sum(axis=0)

        total_steps += num_steps
        # Sander/Mahloujifar evaluate on EMA weights when EMA is enabled.
        eval_model = ema_model if ema_model is not None else model
        test_acc = test(eval_model, test_loader, device)
        epsilon = privacy_engine.get_epsilon(delta=args.delta)
        logger.info(
            f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.2f}%, "
            f"Epsilon: {epsilon:.2f}, Delta: {args.delta}, Steps: {num_steps}, Total Steps: {total_steps}"
        )
        final_test_acc = test_acc

        if epoch % args.ckpt_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, test_acc, ckpt_dir, logger, global_step=total_steps
            )
            if ema_model is not None:
                ema_path = os.path.join(ckpt_dir, f'ema_state_{epoch:06d}.pt')
                torch.save(ema_model.state_dict(), ema_path)
                logger.info(f"Saved EMA state to {ema_path}")

            # Split scores by inclusion mask
            # For paper methods (one-run, f-DP): raw sum
            in_sum = sum_scores[inclusion_mask]
            out_sum = sum_scores[~inclusion_mask]

            # For NDIS: normalized by sqrt(T)
            in_scores_ndis = sum_scores[inclusion_mask] / np.sqrt(total_steps)
            out_scores_ndis = sum_scores[~inclusion_mask] / np.sqrt(total_steps)

            np.savetxt(
                os.path.join(exp_dir, f'in_scores_sum_{epoch:06d}.csv'),
                in_sum, delimiter=',',
            )
            np.savetxt(
                os.path.join(exp_dir, f'out_scores_sum_{epoch:06d}.csv'),
                out_sum, delimiter=',',
            )
            np.savetxt(
                os.path.join(exp_dir, f'in_scores_ndis_{epoch:06d}.csv'),
                in_scores_ndis, delimiter=',',
            )
            np.savetxt(
                os.path.join(exp_dir, f'out_scores_ndis_{epoch:06d}.csv'),
                out_scores_ndis, delimiter=',',
            )
            np.savetxt(
                os.path.join(exp_dir, f'privacy_params_{epoch:06d}.csv'),
                [[epsilon, args.delta]],
                delimiter=',',
                header='current_eps,delta',
                comments='',
            )
            # Save full sum_scores (all canaries) for checkpoint resume
            np.savetxt(
                os.path.join(exp_dir, f'sum_scores_{epoch:06d}.csv'),
                sum_scores, delimiter=',',
            )
            logger.info(
                f"Saved scores at epoch {epoch}: "
                f"in_sum({n_in}), out_sum({n_out}), "
                f"in_ndis({n_in}), out_ndis({n_out})"
            )

        for handler in logging.getLogger().handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
            if hasattr(handler, 'stream') and hasattr(handler.stream, 'flush'):
                handler.stream.flush()

    logger.info("Training complete!")
    save_checkpoint(
        model, optimizer, args.epochs, final_test_acc, ckpt_dir, logger, global_step=total_steps
    )
    logger.info(f"Final log file saved at: {log_file}")

    for handler in logging.getLogger().handlers:
        if hasattr(handler, 'flush'):
            handler.flush()
        if hasattr(handler, 'stream') and hasattr(handler.stream, 'flush'):
            handler.stream.flush()


if __name__ == '__main__':
    main()
