#!/usr/bin/env python3
"""
Training script for differentially private CIFAR-10 classification
using WideResNet with GroupNorm and Weight Standardization.
"""

import sys
import os
import argparse
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm

# ==========================================
# 1. Hyperparameters (SOTA Settings)
# ==========================================
LOGICAL_BATCH_SIZE = 4096     # Target batch size (Paper)
MAX_PHYSICAL_BATCH_SIZE = 128  # GPU limit (128 * 16 = 512 effective images)
AUG_MULTIPLICITY = 16         # K=16 augmentations
MAX_GRAD_NORM = 1.0
EPSILON = 8.0
DELTA = 1e-5
EPOCHS = 140                   # Increase to 100+ for best results
LR = 4.0                      # High LR for large batch
MOMENTUM = 0.0                # No momentum
NOISE_MULTIPLIER = 3.0        # Sigma ~ 3.0 is optimal for BS=4096

# ==========================================
# 2. Architecture: WRN + GroupNorm + WS
# ==========================================
class WSConv2d(nn.Conv2d):
    """Weight Standardized Convolution"""
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.GroupNorm(16, in_planes)
        self.conv1 = WSConv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(16, planes)
        self.conv2 = WSConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                WSConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=4, num_classes=10):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        n = (depth - 4) // 6
        self.conv1 = WSConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(nChannels[0], nChannels[1], n, stride=1)
        self.layer2 = self._make_layer(nChannels[1], nChannels[2], n, stride=2)
        self.layer3 = self._make_layer(nChannels[2], nChannels[3], n, stride=2)
        self.bn1 = nn.GroupNorm(16, nChannels[3])
        self.linear = nn.Linear(nChannels[3], num_classes)

    def _make_layer(self, in_planes, out_planes, nb_layers, stride):
        layers = [WideBasic(in_planes, out_planes, stride)]
        for _ in range(1, nb_layers):
            layers.append(WideBasic(out_planes, out_planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ==========================================
# 3. Logging Setup
# ==========================================
def setup_logging(log_file=None, log_dir='./logs'):
    """Setup logging to both console and file"""
    # Create log directory if it doesn't exist
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    else:
        # Ensure the directory exists for the log file
        log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else '.'
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Create file handler (line buffered for immediate writes)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Set up root logger
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger, log_file

# ==========================================
# 4. Data Loading
# ==========================================
def get_data_loaders(data_dir='./data', logical_batch_size=4096, max_physical_batch_size=128, num_workers=2):
    """Create data loaders for CIFAR-10"""
    # We load "Clean" images. We will augment them K times inside the loop.
    # This saves memory (we don't load 4096*16 images at once).
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # LOADER WITH LOGICAL BATCH SIZE (4096)
    # BatchMemoryManager will handle splitting this into chunks
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=logical_batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )
    
    return train_loader, test_dataset

# ==========================================
# 5. Training Functions
# ==========================================
def train(model, optimizer, train_loader, device, epoch, aug_multiplicity, max_physical_batch_size, logger=None):
    """Training function with augmentation multiplicity and gradient reduction"""
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none') 
    losses = []
    
    # Augmentation transform
    augment_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    
    # BatchMemoryManager splits the 4096 batch into chunks
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=max_physical_batch_size, 
        optimizer=optimizer
    ) as memory_safe_loader:
        
        # Wrap in tqdm
        pbar = tqdm(memory_safe_loader, desc=f"Epoch {epoch}", unit="batch")
        
        for i, (images, labels) in enumerate(pbar):
            # images shape: [B, 3, 32, 32] (Physical Batch)
            
            # --- AUGMENTATION MULTIPLICITY LOGIC ---
            # Expand B -> B*K (e.g., 32 -> 512 with K=16)
            # We apply augmentations manually here on GPU
            images = images.to(device)
            labels = labels.to(device)
            
            # Create K augmentations
            aug_images = []
            for _ in range(aug_multiplicity):
                # Augment requires BCHW
                aug_images.append(augment_transform(images))
            
            # Stack: [K, B, C, H, W] -> [B*K, C, H, W]
            aug_images = torch.stack(aug_images).transpose(0, 1).reshape(-1, 3, 32, 32)
            # Repeat labels: [B] -> [B*K]
            aug_labels = labels.repeat_interleave(aug_multiplicity)
            
            # Forward & Backward
            optimizer.zero_grad()
            outputs = model(aug_images)
            loss_per_sample = criterion(outputs, aug_labels)
            loss = loss_per_sample.mean()
            loss.backward()
            
            # --- GRADIENT REDUCTION (B*K -> B) ---
            # Opacus has computed grads for B*K samples. We reduce them to B.
            for p in model.parameters():
                if hasattr(p, "grad_sample") and p.grad_sample is not None:
                    # Shape: [B*K, ...]
                    gs = p.grad_sample
                    B_K = gs.shape[0]  # Should be B*K
                    feature_shape = gs.shape[1:]
                    
                    # Reshape [B, K, ...]
                    gs_view = gs.view(B_K // aug_multiplicity, aug_multiplicity, *feature_shape)
                    
                    # Average over K
                    gs_avg = gs_view.mean(dim=1)
                    
                    # Replace grad_sample
                    p.grad_sample = gs_avg
            
            # --- OPTIMIZER STEP ---
            # BatchMemoryManager controls this. 
            # If it's a partial batch, Opacus clips and accumulates.
            # If it's the end of logical batch, Opacus noises and updates.
            optimizer.step()
            
            losses.append(loss.item())
            if i % 10 == 0:
                pbar.set_postfix(loss=np.mean(losses[-10:]))
    
    return np.mean(losses)

def test(model, test_loader, device, logger=None):
    """Test function"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# ==========================================
# 6. Checkpoint Saving
# ==========================================
def save_checkpoint(model, optimizer, epoch, test_acc, ckpt_dir, logger):
    """Save model checkpoint to .npz file"""
    ckpt_path = os.path.join(ckpt_dir, f"{epoch:010d}.npz")
    
    # Get model state dict (handle Opacus-wrapped models)
    if hasattr(model, '_module'):
        # Opacus wraps the model, get the underlying model's state dict
        model_state_dict = model._module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    # Get optimizer state dict
    optimizer_state_dict = optimizer.state_dict()
    
    # Convert to numpy arrays for .npz format
    # Store with prefixes to organize model_state_dict and optimizer_state_dict
    npz_dict = {}
    
    # Store model state dict with prefix
    for key, value in model_state_dict.items():
        if isinstance(value, torch.Tensor):
            npz_dict[f'model_state_dict/{key}'] = value.detach().cpu().numpy()
        else:
            npz_dict[f'model_state_dict/{key}'] = value
    
    # Store optimizer state dict with prefix
    for key, value in optimizer_state_dict.items():
        if isinstance(value, torch.Tensor):
            npz_dict[f'optimizer_state_dict/{key}'] = value.detach().cpu().numpy()
        elif isinstance(value, dict):
            # Handle nested dicts in optimizer state (e.g., state for each parameter)
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    npz_dict[f'optimizer_state_dict/{key}/{sub_key}'] = sub_value.detach().cpu().numpy()
                else:
                    npz_dict[f'optimizer_state_dict/{key}/{sub_key}'] = sub_value
        else:
            npz_dict[f'optimizer_state_dict/{key}'] = value
    
    # Save training metadata
    npz_dict['epoch'] = np.array([epoch], dtype=np.int32)
    npz_dict['test_accuracy'] = np.array([test_acc], dtype=np.float32)
    
    np.savez_compressed(ckpt_path, **npz_dict)
    logger.info(f"Checkpoint saved: {ckpt_path} (Epoch {epoch}, Test Accuracy: {test_acc:.2f}%)")

# ==========================================
# 7. Main Function
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Train differentially private CIFAR-10 model')
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory for CIFAR-10 data')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=LR, help='Learning rate')
    parser.add_argument('--logical-batch-size', type=int, default=LOGICAL_BATCH_SIZE, help='Logical batch size')
    parser.add_argument('--max-physical-batch-size', type=int, default=MAX_PHYSICAL_BATCH_SIZE, help='Max physical batch size')
    parser.add_argument('--aug-multiplicity', type=int, default=AUG_MULTIPLICITY, help='Augmentation multiplicity K')
    parser.add_argument('--noise-multiplier', type=float, default=NOISE_MULTIPLIER, help='Noise multiplier (sigma)')
    parser.add_argument('--max-grad-norm', type=float, default=MAX_GRAD_NORM, help='Max gradient norm for clipping')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--log-file', type=str, default=None, help='Path to log file (default: ./logs/train_TIMESTAMP.log)')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Directory for log files')
    parser.add_argument('--ckpt-interval', type=int, default=20, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Setup logging
    logger, log_file = setup_logging(log_file=args.log_file, log_dir=args.log_dir)
    
    # Setup checkpoint directory
    logdir_path = os.path.dirname(log_file) if os.path.dirname(log_file) else args.log_dir
    ckpt_dir = os.path.join(logdir_path, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {ckpt_dir}")
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Device: {device}")
    logger.info("Training parameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Logical batch size: {args.logical_batch_size}")
    logger.info(f"  Max physical batch size: {args.max_physical_batch_size}")
    logger.info(f"  Augmentation multiplicity: {args.aug_multiplicity}")
    logger.info(f"  Noise multiplier: {args.noise_multiplier}")
    logger.info(f"  Max grad norm: {args.max_grad_norm}")
    logger.info(f"  Checkpoint interval: {args.ckpt_interval} epochs")
    
    # Load data
    logger.info("Loading data...")
    train_loader, test_dataset = get_data_loaders(
        data_dir=args.data_dir,
        logical_batch_size=args.logical_batch_size,
        max_physical_batch_size=args.max_physical_batch_size,
        num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024, shuffle=False, num_workers=args.num_workers
    )
    
    # Create model
    logger.info("Creating model...")
    model = WideResNet(depth=16, widen_factor=4).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM)
    
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
    
    # Training loop
    logger.info("Starting training...")
    # Flush all handlers to ensure immediate output
    for handler in logging.getLogger().handlers:
        handler.flush()
        if hasattr(handler, 'stream') and hasattr(handler.stream, 'flush'):
            handler.stream.flush()
    
    final_test_acc = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            model, optimizer, train_loader, device, epoch,
            args.aug_multiplicity, args.max_physical_batch_size, logger=logger
        )
        test_acc = test(model, test_loader, device, logger=logger)
        
        # Get current privacy budget (epsilon)
        epsilon = privacy_engine.get_epsilon(delta=DELTA)
        
        logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Epsilon: {epsilon:.2f}, Delta: {DELTA}")
        final_test_acc = test_acc  # Store for final checkpoint
        
        # Save checkpoint every N epochs
        if epoch % args.ckpt_interval == 0:
            save_checkpoint(model, optimizer, epoch, test_acc, ckpt_dir, logger)
        
        # Flush after each epoch
        for handler in logging.getLogger().handlers:
            handler.flush()
            if hasattr(handler, 'stream') and hasattr(handler.stream, 'flush'):
                handler.stream.flush()
    
    logger.info("Training complete!")
    
    # Save final model checkpoint (if not already saved at the last interval)
    if args.epochs % args.ckpt_interval != 0:
        save_checkpoint(model, optimizer, args.epochs, final_test_acc, ckpt_dir, logger)
    
    logger.info(f"Final log file saved at: {log_file}")
    
    # Final flush
    for handler in logging.getLogger().handlers:
        handler.flush()
        if hasattr(handler, 'stream') and hasattr(handler.stream, 'flush'):
            handler.stream.flush()

if __name__ == '__main__':
    main()


