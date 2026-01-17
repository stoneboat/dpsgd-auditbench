import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm

def train(model, optimizer, train_loader, device, epoch, aug_multiplicity, max_physical_batch_size, logical_batch_size):
    """Training function with augmentation multiplicity and gradient reduction
    
    Args:
        model: The model to train
        optimizer: The optimizer
        train_loader: Training data loader
        device: Device to use
        epoch: Current epoch number
        aug_multiplicity: Number of augmentations per sample (K)
        max_physical_batch_size: Maximum physical batch size for BatchMemoryManager
        logical_batch_size: Logical batch size (required for counting logical batches)
    """
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none') 
    losses = []
    
    # Augmentation transform
    augment_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    
    # BatchMemoryManager splits the logical batch into chunks
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=max_physical_batch_size, 
        optimizer=optimizer
    ) as memory_safe_loader:
        
        # Wrap in tqdm
        pbar = tqdm(memory_safe_loader, desc=f"Epoch {epoch}", unit="batch")
        
        num_logical_steps = 0  # Count logical batches, not physical batches
        samples_in_current_logical_batch = 0  # Track samples processed in current logical batch
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

            loss = loss_per_sample.mean() * aug_multiplicity
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
            
            # Track samples processed for logical batch counting
            # images.shape[0] is the physical batch size (before augmentation)
            physical_batch_size = images.shape[0]
            samples_in_current_logical_batch += physical_batch_size
            
            # When we've processed a full logical batch, count it as one step
            if samples_in_current_logical_batch >= logical_batch_size:
                num_logical_steps += 1
                samples_in_current_logical_batch = 0  # Reset for next logical batch
            
            losses.append(loss.item())
            if i % 10 == 0:
                pbar.set_postfix(loss=np.mean(losses[-10:]))
    
    return np.mean(losses), num_logical_steps

def test(model, test_loader, device):
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