#!/usr/bin/env python3
"""
Training script for differentially private CIFAR-10 classification with privacy auditing.
Uses WideResNet with GroupNorm and Weight Standardization, with canary-based auditing.
"""

import sys
import os
import json
import secrets
import numpy as np
import argparse
import logging
import torch
from opacus import PrivacyEngine
import torch.optim as optim

# Add the src directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '../'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)

from utils import setup_logging, save_checkpoint, find_latest_checkpoint, load_checkpoint
from dataset import get_auditable_data_loaders
from network_arch import WideResNet
from train import train, test

# ==========================================
# Default Hyperparameters
# ==========================================
DEFAULT_LOGICAL_BATCH_SIZE = 4096
DEFAULT_MAX_PHYSICAL_BATCH_SIZE = 128
DEFAULT_AUG_MULTIPLICITY = 16
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_EPSILON = 8.0
DEFAULT_DELTA = 1e-5
DEFAULT_EPOCHS = 200
DEFAULT_LR = 4.0
DEFAULT_MOMENTUM = 0.0
DEFAULT_NOISE_MULTIPLIER = 3.0
DEFAULT_CKPT_INTERVAL = 20
DEFAULT_CANARY_COUNT = 5000
DEFAULT_PKEEP = 0.5

def main():
    parser = argparse.ArgumentParser(description='Train differentially private CIFAR-10 model with privacy auditing')
    
    # Hyperparameters
    parser.add_argument('--logical-batch-size', type=int, default=DEFAULT_LOGICAL_BATCH_SIZE, help='Logical batch size')
    parser.add_argument('--max-physical-batch-size', type=int, default=DEFAULT_MAX_PHYSICAL_BATCH_SIZE, help='Max physical batch size')
    parser.add_argument('--aug-multiplicity', type=int, default=DEFAULT_AUG_MULTIPLICITY, help='Augmentation multiplicity K')
    parser.add_argument('--max-grad-norm', type=float, default=DEFAULT_MAX_GRAD_NORM, help='Max gradient norm for clipping')
    parser.add_argument('--epsilon', type=float, default=DEFAULT_EPSILON, help='Target epsilon (privacy budget)')
    parser.add_argument('--delta', type=float, default=DEFAULT_DELTA, help='Delta for (epsilon, delta)-DP')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=DEFAULT_MOMENTUM, help='Momentum')
    parser.add_argument('--noise-multiplier', type=float, default=DEFAULT_NOISE_MULTIPLIER, help='Noise multiplier (sigma)')
    parser.add_argument('--ckpt-interval', type=int, default=DEFAULT_CKPT_INTERVAL, help='Save checkpoint every N epochs')
    
    # Experiment parameters
    parser.add_argument('--canary-count', type=int, default=DEFAULT_CANARY_COUNT, help='Number of canaries')
    parser.add_argument('--pkeep', type=float, default=DEFAULT_PKEEP, help='Probability of including each canary')
    parser.add_argument('--database-seed', type=str, default=None, help='Random seed for dataset as string (if None, generates random 128-bit seed)')
    
    # Paths
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory for CIFAR-10 data')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Directory for log files')
    parser.add_argument('--log-file', type=str, default=None, help='Path to log file (default: ./logs/train_TIMESTAMP.log)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    logger, log_file = setup_logging(log_file=args.log_file, log_dir=args.log_dir)
    
    # Handle database seed
    if args.database_seed is None:
        DATABSEED = secrets.randbits(128)
        logger.info(f"Generated random 128-bit seed: {DATABSEED}")
    else:
        # Convert string to int (Python's int is arbitrary precision, so it can handle 128-bit integers)
        try:
            DATABSEED = int(args.database_seed)
        except ValueError:
            logger.error(f"Invalid database-seed: '{args.database_seed}'. Must be a valid integer.")
            sys.exit(1)
        logger.info(f"Using provided database seed: {DATABSEED}")
    
    # Create experiment directory
    exp_dir = os.path.join(args.data_dir, f"mislabeled-canaries-{DATABSEED}-{args.canary_count}-{args.pkeep}-cifar10")
    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Experiment directory: {exp_dir}")
    
    # Create checkpoint directory
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {ckpt_dir}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Run experiment on device: {device}")
    
    # Store hyperparameters
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
        'database_seed': DATABSEED
    }
    
    # Save hyperparameters
    hparams_path = os.path.join(exp_dir, 'hparams.json')
    with open(hparams_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Hyperparameters saved to: {hparams_path}")
    
    # Load data
    logger.info("Loading data...")
    train_dataset, test_dataset = get_auditable_data_loaders(
        data_dir=args.data_dir,
        canary_count=args.canary_count,
        seed=DATABSEED,
        pkeep=args.pkeep
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024, shuffle=False, num_workers=args.num_workers
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.logical_batch_size, shuffle=True, 
        num_workers=args.num_workers, drop_last=True
    )
    
    # Create model
    logger.info("Creating model...")
    model = WideResNet(depth=16, widen_factor=4).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    # Setup privacy engine
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
    
    checkpoint_result = find_latest_checkpoint(ckpt_dir)
    if checkpoint_result is not None:
        checkpoint_path, checkpoint_epoch = checkpoint_result
        logger.info(f"Loading checkpoint '{checkpoint_path}' (epoch {checkpoint_epoch})...")
        
        # Load model and optimizer state
        loaded_epoch, loaded_global_step = load_checkpoint(checkpoint_path, model, optimizer, device, logger)
        start_epoch = loaded_epoch + 1
        
        # Recover privacy state
        steps_per_epoch = len(train_loader) 
        sample_rate = 1 / len(train_loader)
        
        # Update privacy accountant history
        if loaded_global_step is not None:
            privacy_engine.accountant.history.append((args.noise_multiplier, sample_rate, loaded_global_step))
            total_steps = loaded_global_step
            logger.info(f"Resumed from Epoch {start_epoch}")
            logger.info(f"Privacy Accountant updated with {loaded_global_step} past steps.")
            
            # Verify epsilon
            current_eps = privacy_engine.get_epsilon(args.delta)
            logger.info(f"Current Cumulative Epsilon: {current_eps:.2f}")
        else:
            logger.warning("Checkpoint does not contain global_step. Starting from step 0.")
    else:
        logger.info("No checkpoint found. Starting from scratch.")
    
    # Training loop
    logger.info("Starting training...")
    logger.info("Training parameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Logical batch size: {args.logical_batch_size}")
    logger.info(f"  Max physical batch size: {args.max_physical_batch_size}")
    logger.info(f"  Augmentation multiplicity: {args.aug_multiplicity}")
    logger.info(f"  Noise multiplier: {args.noise_multiplier}")
    logger.info(f"  Max grad norm: {args.max_grad_norm}")
    logger.info(f"  Canary count: {args.canary_count}")
    logger.info(f"  P(keep): {args.pkeep}")
    logger.info(f"  Checkpoint interval: {args.ckpt_interval} epochs")
    
    # Flush handlers
    for handler in logging.getLogger().handlers:
        handler.flush()
        if hasattr(handler, 'stream') and hasattr(handler.stream, 'flush'):
            handler.stream.flush()
    
    final_test_acc = None
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, num_steps = train(
            model, optimizer, train_loader, device, epoch, 
            args.aug_multiplicity, args.max_physical_batch_size, args.logical_batch_size
        )
        total_steps += num_steps
        test_acc = test(model, test_loader, device)
        
        # Get current privacy budget (epsilon)
        epsilon = privacy_engine.get_epsilon(delta=args.delta)
        
        logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Epsilon: {epsilon:.2f}, Delta: {args.delta}, Steps: {num_steps}, Total Steps: {total_steps}")
        final_test_acc = test_acc
        
        # Save checkpoint every N epochs
        if epoch % args.ckpt_interval == 0:
            save_checkpoint(model, optimizer, epoch, test_acc, ckpt_dir, logger, global_step=total_steps)
        
        # Flush after each epoch
        for handler in logging.getLogger().handlers:
            handler.flush()
            if hasattr(handler, 'stream') and hasattr(handler.stream, 'flush'):
                handler.stream.flush()
    
    logger.info("Training complete!")
    
    # Save final checkpoint (if not already saved at the last interval)
    if args.epochs % args.ckpt_interval != 0:
        save_checkpoint(model, optimizer, args.epochs, final_test_acc, ckpt_dir, logger, global_step=total_steps)
    
    logger.info(f"Final log file saved at: {log_file}")
    
    # Final flush
    for handler in logging.getLogger().handlers:
        handler.flush()
        if hasattr(handler, 'stream') and hasattr(handler.stream, 'flush'):
            handler.stream.flush()

if __name__ == '__main__':
    main()
