import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm


def make_dirac_canary_like(params, param_index=0, flat_index=0, device=None, dtype=None):
    """Create a dirac canary vector matching params (1 at a single coord, else 0)."""
    canary_tensors = []
    for idx, p in enumerate(params):
        t = torch.zeros_like(p, device=device or p.device, dtype=dtype or p.dtype)
        if idx == param_index:
            flat = t.view(-1)
            if flat_index < 0 or flat_index >= flat.numel():
                raise ValueError("flat_index out of range for selected parameter.")
            flat[flat_index] = 1.0
        canary_tensors.append(t)
    return canary_tensors


def make_random_unit_like(params, generator=None, device=None, dtype=None):
    """Create a random unit vector matching params."""
    canary_tensors = []
    total_sq = None
    for p in params:
        t = torch.randn_like(p, device=device or p.device, dtype=dtype or p.dtype, generator=generator)
        total_sq = (t * t).sum() if total_sq is None else total_sq + (t * t).sum()
        canary_tensors.append(t)
    norm = torch.sqrt(total_sq)
    if not torch.isfinite(norm) or norm.item() == 0.0:
        raise ValueError("Random canary has zero or non-finite norm.")
    canary_tensors = [t / norm for t in canary_tensors]
    return canary_tensors


def _normalize_canary_tensors(canary_tensors, device):
    total_sq = None
    normed = []
    for t in canary_tensors:
        tt = t.to(device=device)
        total_sq = (tt * tt).sum() if total_sq is None else total_sq + (tt * tt).sum()
        normed.append(tt)
    norm = torch.sqrt(total_sq)
    if not torch.isfinite(norm) or norm.item() == 0.0:
        raise ValueError("Canary direction has zero or non-finite norm.")
    normed = [t / norm for t in normed]
    return normed, norm.item()


def _whitebox_dp_step(
    optimizer,
    canary_scale,
    canary_prob,
    scores,
    include_flags,
    canary_dirac,
):
    if not hasattr(optimizer, "clip_and_accumulate"):
        return optimizer.step()

    if optimizer.grad_samples is None or len(optimizer.grad_samples) == 0:
        return optimizer.step()

    optimizer.clip_and_accumulate()

    if hasattr(optimizer, "_check_skip_next_step") and optimizer._check_skip_next_step():
        optimizer._is_last_step_skipped = True
        return None

    optimizer._is_last_step_skipped = False

    base_summed = [p.summed_grad.detach().clone() for p in optimizer.params]
    include = False
    if canary_dirac is not None and canary_prob is not None and canary_prob > 0.0:
        device = canary_dirac["device"]
        include_mask = torch.rand(
            (canary_dirac["num_canaries"],), device=device
        ) < canary_prob
        if include_mask.any():
            for param_idx, p in enumerate(optimizer.params):
                if param_idx not in canary_dirac["by_param"]:
                    continue
                flat_indices, canary_ids = canary_dirac["by_param"][param_idx]
                sel = include_mask[canary_ids]
                if sel.any():
                    flat = p.summed_grad.view(-1)
                    flat.index_add_(
                        0,
                        flat_indices[sel],
                        torch.full(
                            (int(sel.sum().item()),),
                            canary_scale,
                            device=flat.device,
                            dtype=flat.dtype,
                        ),
                    )
        include = True

    optimizer.add_noise()

    if scores is not None and canary_dirac is not None:
        step_scores = torch.empty(
            (canary_dirac["num_canaries"],), device=canary_dirac["device"]
        )
        for param_idx, p in enumerate(optimizer.params):
            if param_idx not in canary_dirac["by_param"]:
                continue
            flat_indices, canary_ids = canary_dirac["by_param"][param_idx]
            delta = (p.grad - base_summed[param_idx]).view(-1)
            step_scores[canary_ids] = delta[flat_indices]
        scores.append(step_scores.detach().cpu().tolist())
        if include_flags is not None:
            include_flags.append(include_mask.detach().cpu().tolist())

    optimizer.scale_grad()
    if optimizer.step_hook:
        optimizer.step_hook(optimizer)

    return optimizer.original_optimizer.step()

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
    normalize_transform = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    )
    
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
            aug_images = normalize_transform(aug_images)
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


def train_whitebox(
    model,
    optimizer,
    train_loader,
    device,
    epoch,
    aug_multiplicity,
    max_physical_batch_size,
    logical_batch_size,
    *,
    canary_dirac_indices=None,
    canary_prob=None,
    canary_scale=None,
    return_scores=False,
    return_include_flags=False,
):
    """Training function for auditing in whitebox model.

    canary_dirac_indices: list of (param_index, flat_index) for dirac canaries.
    canary_prob: probability q of injecting the canary per logical step.
    canary_scale: magnitude (mu) of the canary; defaults to optimizer.max_grad_norm.
    """
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none') 
    losses = []

    scores = [] if return_scores else None
    include_flags = [] if return_include_flags else None
    canary_dirac = None
    if canary_dirac_indices is not None:
        by_param = {}
        param_numel = [p.numel() for p in model.parameters()]
        for canary_id, (param_idx, flat_idx) in enumerate(canary_dirac_indices):
            if param_idx < 0:
                raise ValueError("param_idx must be non-negative.")
            if flat_idx < 0:
                raise ValueError("flat_idx must be non-negative.")
            if param_idx >= len(param_numel):
                raise ValueError(f"param_idx {param_idx} out of range (num params={len(param_numel)}).")
            if flat_idx >= param_numel[param_idx]:
                raise ValueError(
                    f"flat_idx {flat_idx} out of range for param {param_idx} (numel={param_numel[param_idx]})."
                )
            by_param.setdefault(param_idx, {"flat": [], "ids": []})
            by_param[param_idx]["flat"].append(int(flat_idx))
            by_param[param_idx]["ids"].append(int(canary_id))

        by_param_t = {}
        for param_idx, data in by_param.items():
            by_param_t[param_idx] = (
                torch.tensor(data["flat"], device=device, dtype=torch.long),
                torch.tensor(data["ids"], device=device, dtype=torch.long),
            )

        canary_dirac = {
            "by_param": by_param_t,
            "num_canaries": len(canary_dirac_indices),
            "device": device,
        }

    if canary_prob is None:
        canary_prob = 1.0 / max(1, len(train_loader))
    if canary_scale is None and hasattr(optimizer, "max_grad_norm"):
        canary_scale = float(optimizer.max_grad_norm)
    if canary_scale is None:
        canary_scale = 1.0
    
    # Augmentation transform
    augment_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    normalize_transform = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    )
    
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
            aug_images = normalize_transform(aug_images)
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
            _whitebox_dp_step(
                optimizer,
                canary_scale=canary_scale,
                canary_prob=canary_prob,
                scores=scores,
                include_flags=include_flags,
                canary_dirac=canary_dirac,
            )
            
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
    
    if return_scores:
        num_logical_steps = len(scores)
        if return_include_flags:
            return np.mean(losses), num_logical_steps, scores, include_flags
        return np.mean(losses), num_logical_steps, scores
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
