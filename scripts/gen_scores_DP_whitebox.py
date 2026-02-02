#!/usr/bin/env python3
"""
Generate in/out audit scores for white-box DP-SGD auditing.
Trains a DP model with dirac canaries and saves in_scores, out_scores, and
privacy_params at each checkpoint interval (same logic as train_auditable_DP_model_whitebox.ipynb).
"""

import sys
import os
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
from classifier.white_box_dp_sgd import sample_gaussian

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
DEFAULT_LR = 4.0
DEFAULT_MOMENTUM = 0.0
DEFAULT_NOISE_MULTIPLIER = 3.0
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
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--momentum', type=float, default=DEFAULT_MOMENTUM)
    parser.add_argument('--noise-multiplier', type=float, default=DEFAULT_NOISE_MULTIPLIER)
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
        'lr': args.lr,
        'momentum': args.momentum,
        'noise_multiplier': args.noise_multiplier,
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
    logger.info("Setting up privacy engine...")
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )

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
            (args.noise_multiplier, sample_rate, loaded_global_step)
        )
        logger.info(f"Resumed from Epoch {start_epoch}")
        logger.info(f"Privacy Accountant updated with {loaded_global_step} past steps.")
        total_steps = loaded_global_step if loaded_global_step is not None else 0

        in_scores_path = os.path.join(exp_dir, f'in_scores_{checkpoint_epoch:06d}.csv')
        out_scores_path = os.path.join(exp_dir, f'out_scores_{checkpoint_epoch:06d}.csv')
        if os.path.isfile(in_scores_path) and total_steps > 0:
            in_scores_loaded = np.loadtxt(in_scores_path, delimiter=',')
            sum_scores = in_scores_loaded * total_steps
            logger.info(
                f"Loaded in_scores for epoch {checkpoint_epoch} -> restored sum_scores (total_steps={total_steps})"
            )
        if os.path.isfile(out_scores_path):
            out_scores_loaded = np.loadtxt(out_scores_path, delimiter=',')
            logger.info(f"Loaded out_scores for epoch {checkpoint_epoch} (same epoch as checkpoint)")

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

    logger.info("Starting training...")
    logger.info(f"  Epochs: {args.epochs}, LR: {args.lr}, Momentum: {args.momentum}")
    logger.info(f"  Logical batch size: {args.logical_batch_size}")
    logger.info(f"  Max physical batch size: {args.max_physical_batch_size}")
    logger.info(f"  Aug multiplicity: {args.aug_multiplicity}")
    logger.info(f"  Noise multiplier: {args.noise_multiplier}")
    logger.info(f"  Canary count: {args.canary_count}, P(keep): {args.pkeep}")
    logger.info(f"  Checkpoint interval: {args.ckpt_interval} epochs")

    final_test_acc = None
    canary_prob = 1.0 / len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
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
            canary_prob=canary_prob,
            return_scores=True,
        )
        scores = np.asarray(scores)
        assert scores.shape[0] == num_steps, (
            f"scores.shape[0] = {scores.shape[0]} != num_steps = {num_steps}"
        )
        if sum_scores is None:
            sum_scores = scores.sum(axis=0)
        else:
            sum_scores = sum_scores + scores.sum(axis=0)

        total_steps += num_steps
        test_acc = test(model, test_loader, device)
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
            in_scores = sum_scores / total_steps
            out_canary_observations = sample_gaussian(
                total_steps, args.canary_count, args.noise_multiplier, rng
            )
            out_scores = out_canary_observations.sum(axis=1) / total_steps
            np.savetxt(
                os.path.join(exp_dir, f'out_scores_{epoch:06d}.csv'),
                out_scores,
                delimiter=',',
            )
            np.savetxt(
                os.path.join(exp_dir, f'in_scores_{epoch:06d}.csv'),
                in_scores,
                delimiter=',',
            )
            np.savetxt(
                os.path.join(exp_dir, f'privacy_params_{epoch:06d}.csv'),
                [[epsilon, args.delta]],
                delimiter=',',
                header='current_eps,delta',
                comments='',
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
