import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm


_PER_IMAGE_AUG = transforms_v2.Compose([
    transforms_v2.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms_v2.RandomHorizontalFlip(p=0.5),
])


def _augment_per_image(images):
    # torchvision v2 transforms applied to a batched tensor still share one
    # random state across the batch; vmap over the batch dim to get independent
    # augmentations per image (Sander et al. / Mahloujifar et al. recipe).
    return torch.stack([_PER_IMAGE_AUG(img) for img in images])


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

    include_mask = None
    if canary_dirac is not None and canary_prob is not None and canary_prob > 0.0:
        device = canary_dirac["device"]
        # Per-step sampling: each canary is sampled with prob canary_prob
        include_mask = torch.rand(
            (canary_dirac["num_canaries"],), device=device
        ) < canary_prob

        # Only inject IN canaries: AND per-step sampling with membership mask
        membership_mask = canary_dirac.get("inclusion_mask")
        if membership_mask is not None:
            inject_mask = include_mask & membership_mask
        else:
            inject_mask = include_mask

        if inject_mask.any():
            for param_idx, p in enumerate(optimizer.params):
                if param_idx not in canary_dirac["by_param"]:
                    continue
                flat_indices, canary_ids = canary_dirac["by_param"][param_idx]
                sel = inject_mask[canary_ids]
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

    optimizer.add_noise()

    # Record scores at ALL canary coordinates (both in and out).
    # The inclusion_mask (saved separately) determines which is which at analysis time.
    if scores is not None and canary_dirac is not None:
        step_scores = torch.empty(
            (canary_dirac["num_canaries"],), device=canary_dirac["device"]
        )
        for param_idx, p in enumerate(optimizer.params):
            if param_idx not in canary_dirac["by_param"]:
                continue
            flat_indices, canary_ids = canary_dirac["by_param"][param_idx]
            noised_grad = p.grad.view(-1)
            step_scores[canary_ids] = noised_grad[flat_indices]
        scores.append(step_scores.detach().cpu().tolist())
        if include_flags is not None and include_mask is not None:
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
    # Sander et al. (2023) / Mahloujifar et al. (2024) augmentation recipe.
    # torchvision v1 Compose on a batched tensor applies the SAME random params
    # to every image in the batch; we want per-image randomness, so we apply per
    # image inside the K loop below via _augment_per_image().
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
                aug_images.append(_augment_per_image(images))
            
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

                    # Sum over K augs. instead of mean
                    gs_sum = gs_view.sum(dim=1)

                    # Replace grad_sample
                    p.grad_sample = gs_sum

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
    canary_inclusion_mask=None,
    canary_prob=None,
    canary_scale=None,
    return_scores=False,
    return_include_flags=False,
    ema_model=None,
    ema_decay=0.9999,
    ema_step_offset=0,
    max_logical_steps=None,
):
    """Training function for auditing in whitebox model.

    canary_dirac_indices: list of (param_index, flat_index) for dirac canaries.
    canary_inclusion_mask: boolean array of length num_canaries. True = IN (inject signal),
        False = OUT (no injection, pure noise). Scores are recorded for ALL canaries.
        If None, all canaries are injected (legacy behavior).
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

        inclusion_mask_t = None
        if canary_inclusion_mask is not None:
            inclusion_mask_t = torch.as_tensor(
                canary_inclusion_mask, device=device, dtype=torch.bool
            )
            assert inclusion_mask_t.shape[0] == len(canary_dirac_indices), (
                f"canary_inclusion_mask length {inclusion_mask_t.shape[0]} != "
                f"num canaries {len(canary_dirac_indices)}"
            )

        canary_dirac = {
            "by_param": by_param_t,
            "num_canaries": len(canary_dirac_indices),
            "device": device,
            "inclusion_mask": inclusion_mask_t,
        }

    if canary_prob is None:
        canary_prob = 1.0 / max(1, len(train_loader))
    if canary_scale is None and hasattr(optimizer, "max_grad_norm"):
        canary_scale = float(optimizer.max_grad_norm)
    if canary_scale is None:
        canary_scale = 1.0
    
    # Augmentation transform
    # Sander et al. (2023) / Mahloujifar et al. (2024) augmentation recipe.
    # torchvision v1 Compose on a batched tensor applies the SAME random params
    # to every image in the batch; we want per-image randomness, so we apply per
    # image inside the K loop below via _augment_per_image().
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
                aug_images.append(_augment_per_image(images))
            
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

                    # Sum over K augs (see the train() function above for the full
                    # rationale). This makes per-logical-sample grad norm ~K * g_i so
                    # clipping at C=1 reliably saturates and uses the full DP budget.
                    gs_sum = gs_view.sum(dim=1)

                    # Replace grad_sample
                    p.grad_sample = gs_sum

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

                if ema_model is not None:
                    # TIMM-style warmup so the EMA isn't dominated by the random init
                    # over short training runs (Sander et al. recipe: 2500 steps).
                    global_step = ema_step_offset + num_logical_steps
                    effective_decay = min(ema_decay, (1.0 + global_step) / (10.0 + global_step))
                    with torch.no_grad():
                        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                            p_ema.data.mul_(effective_decay).add_(p.data, alpha=1.0 - effective_decay)
                        for b_ema, b in zip(ema_model.buffers(), model.buffers()):
                            b_ema.data.copy_(b.data)

                if max_logical_steps is not None and num_logical_steps >= max_logical_steps:
                    break

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
