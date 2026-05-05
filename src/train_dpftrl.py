"""Centralized DP-FTRL training with tree-aggregated Gaussian noise.

Each step processes one mini-batch and produces a "leaf" gradient
g_t = (sum of clipped per-sample gradients) + (sum of canary signals due at t).
Leaves enter a streaming binary tree (Honaker construction); we maintain
the noisy cumulative gradient

    G_t_noisy  =  clean_G_t  +  (sum of fresh noises at canonical nodes for [0..t])

and the FTRL update at step t is

    theta_t  =  theta_init  -  lr * G_t_noisy.

Audit setup:
  * Each canary i has a fixed dirac direction (`canary_dirac_indices[i]` =
    a (param_idx, flat_idx) pair) and a fixed leaf t_star_i in [0, T).
  * If `inclusion_mask[i]` is True, the canary's signal `+canary_scale` is
    added to the leaf gradient at step t_star_i.
  * The SNR-optimal score per canary is

        score_i  =  sum over a in ancestors(t_star_i) of
                       (clean_partial_sum_at_a + noise_at_a)[canary_coord_i]

    To compute it cheaply, we record:
      - per leaf t: the *per-canary* projection of leaf_grad at canary i's
        coord (a [T, m] float array, ~50MB at T=2500, m=5000),
      - per newly-created tree node (level, idx): the per-canary projection
        of the fresh noise tensor at canary i's coord (~log T floats per
        canary, ~hundreds of KB).
    At the end, we sum ancestor partial sums + ancestor noises per canary.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from train import _augment_per_image  # reuse per-image augmentation

_logger = logging.getLogger(__name__)
# Throttle pre-clip grad-norm telemetry. Logged from _clip_and_sum_grad_samples,
# which is called once per physical batch. With ~32 physical batches per leaf,
# N=50 ~= once or twice per leaf. Set to 0 to disable.
_GRAD_NORM_LOG_EVERY = 50
_grad_norm_call_count = [0]


# ---------------------------------------------------------------------------
# Canonical decomposition (matches whitebox_auditing.tree_mechanism, repeated
# here as a set for fast diff operations).
# ---------------------------------------------------------------------------


def _bit_decomposition_set(n: int) -> set:
    """Set of (level, idx) canonical nodes covering [0..n-1]; empty for n<=0."""
    out: set = set()
    if n <= 0:
        return out
    offset = 0
    msb = n.bit_length() - 1
    for level in range(msb, -1, -1):
        if n & (1 << level):
            out.add((level, offset >> level))
            offset += 1 << level
    return out


# ---------------------------------------------------------------------------
# Per-sample clipping helper (replaces Opacus's optimizer-level clipping).
# ---------------------------------------------------------------------------


def _clip_and_sum_grad_samples(
    params: List[torch.nn.Parameter], max_grad_norm: float
) -> List[torch.Tensor]:
    """Per-sample L2 clip then sum across the batch.

    Each parameter has `p.grad_sample` of shape [B, ...] (set by Opacus's
    GradSampleModule). Returns one tensor per parameter with `p.shape`,
    holding sum_{b}(min(1, C/||g_b||) * g_b).
    """
    per_sample_sq = None
    for p in params:
        gs = p.grad_sample
        if gs is None:
            continue
        flat = gs.view(gs.shape[0], -1)
        sq = (flat * flat).sum(dim=1)
        per_sample_sq = sq if per_sample_sq is None else per_sample_sq + sq
    norms = per_sample_sq.clamp_min(1e-12).sqrt()

    # Pre-clip telemetry (throttled): tells us whether C is saturating or wasted.
    if _GRAD_NORM_LOG_EVERY > 0:
        _grad_norm_call_count[0] += 1
        if _grad_norm_call_count[0] % _GRAD_NORM_LOG_EVERY == 1:
            n_np = norms.detach().cpu().numpy()
            _logger.info(
                "grad_norms B=%d C=%.2f mean=%.4f median=%.4f p90=%.4f p99=%.4f "
                "frac_>=C=%.3f frac_<0.5C=%.3f",
                int(n_np.shape[0]), float(max_grad_norm),
                float(np.mean(n_np)),
                float(np.median(n_np)),
                float(np.percentile(n_np, 90)),
                float(np.percentile(n_np, 99)),
                float(np.mean(n_np >= max_grad_norm)),
                float(np.mean(n_np < 0.5 * max_grad_norm)),
            )

    scaling = (max_grad_norm / norms).clamp_max(1.0)   # [B]

    summed: List[torch.Tensor] = []
    for p in params:
        gs = p.grad_sample
        if gs is None:
            summed.append(torch.zeros_like(p))
            continue
        weighted = gs * scaling.view([gs.shape[0]] + [1] * (gs.ndim - 1))
        summed.append(weighted.sum(dim=0))
        p.grad_sample = None
    return summed


# ---------------------------------------------------------------------------
# DP-FTRL state -- streaming Honaker tree + audit recording.
# ---------------------------------------------------------------------------
class DPFTRLState:
    def __init__(
        self,
        params: List[torch.nn.Parameter],
        device: torch.device,
        sigma_node: float,
        target_steps: int,
        canary_dirac_indices: List[Tuple[int, int]],
        canary_leaf_assignment: np.ndarray,
        canary_inclusion_mask: np.ndarray,
        canary_scale: float,
    ) -> None:
        self.params = params
        self.device = device
        self.sigma_node = float(sigma_node)
        self.T = int(target_steps)

        # Honaker streaming state -- full-tensor running sums.
        self.clean_G = [torch.zeros_like(p) for p in params]
        self.canonical_noise = [torch.zeros_like(p) for p in params]
        # Stored noise per active canonical node, so we can subtract on roll-up.
        self.noise_per_active: Dict[Tuple[int, int], List[torch.Tensor]] = {}

        # Canary metadata.
        self.dirac = canary_dirac_indices
        self.m = len(canary_dirac_indices)
        self.leaf_of = np.asarray(canary_leaf_assignment, dtype=np.int64)
        self.in_mask = np.asarray(canary_inclusion_mask, dtype=bool)
        self.canary_scale = float(canary_scale)

        self._canaries_at_leaf: Dict[int, List[int]] = {}
        for i, t in enumerate(self.leaf_of):
            self._canaries_at_leaf.setdefault(int(t), []).append(i)

        self._coords_by_param: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        by_param: Dict[int, List[Tuple[int, int]]] = {}
        for i, (p_idx, flat_idx) in enumerate(canary_dirac_indices):
            by_param.setdefault(p_idx, []).append((flat_idx, i))
        for p_idx, lst in by_param.items():
            flats = torch.tensor([f for f, _ in lst], device=device, dtype=torch.long)
            ids   = torch.tensor([i for _, i in lst], device=device, dtype=torch.long)
            self._coords_by_param[p_idx] = (flats, ids)

        self.leaf_proj  = np.zeros((self.T, self.m), dtype=np.float32)
        self.node_noise_proj: Dict[Tuple[int, int], np.ndarray] = {}

    # ----------------------------------------------------------------------
    def _project_to_canaries(self, tensors: List[torch.Tensor]) -> np.ndarray:
        """Per-canary projection: tensor[(p_i, flat_i)] for each canary i."""
        out = torch.zeros(self.m, device=self.device)
        for p_idx, (flats, ids) in self._coords_by_param.items():
            out.index_add_(0, ids, tensors[p_idx].view(-1)[flats])
        return out.detach().cpu().numpy().astype(np.float32)

    def _inject_canaries(self, leaf_grad: List[torch.Tensor], t: int) -> None:
        """Add `canary_scale * e_{c_i}` to leaf_grad for each in-canary with t_star == t."""
        ids_here = self._canaries_at_leaf.get(t, ())
        for i in ids_here:
            if self.in_mask[i]:
                p_idx, flat_idx = self.dirac[i]
                leaf_grad[p_idx].view(-1)[flat_idx] += self.canary_scale

    def step(self, leaf_grad: List[torch.Tensor], t: int) -> List[torch.Tensor]:
        self._inject_canaries(leaf_grad, t)
        self.leaf_proj[t, :] = self._project_to_canaries(leaf_grad)
        for c, g in zip(self.clean_G, leaf_grad):
            c.add_(g)

        added = _bit_decomposition_set(t + 1) - _bit_decomposition_set(t)
        removed = _bit_decomposition_set(t) - _bit_decomposition_set(t + 1)
        for node in removed:
            noise = self.noise_per_active.pop(node)
            for canon, n_t in zip(self.canonical_noise, noise):
                canon.sub_(n_t)
        for node in sorted(added):
            noise = [torch.randn_like(p) * self.sigma_node for p in self.params]
            self.noise_per_active[node] = noise
            for canon, n_t in zip(self.canonical_noise, noise):
                canon.add_(n_t)
            self.node_noise_proj[node] = self._project_to_canaries(noise)

        noisy_G = [c + n for c, n in zip(self.clean_G, self.canonical_noise)]
        return noisy_G

    # ----------------------------------------------------------------------
    def compute_optimal_scores(self, leaves_done: int) -> np.ndarray:
        """SNR-optimal audit score per canary, post-training.

        score_i = sum over a in ancestors(t_star_i) of
                  (clean_partial_sum_at_a[c_i] + noise_at_a[c_i])
        """
        # Cumulative per-canary leaf projections for O(1) range sums.
        cum = np.zeros((leaves_done + 1, self.m), dtype=np.float64)
        cum[1:] = np.cumsum(self.leaf_proj[:leaves_done].astype(np.float64), axis=0)

        # The tree is over T_pow2 = next_pow2(T) leaves; a single canary's
        # ancestor list spans levels 0..ceil(log2 T_pow2).
        T_pow2 = 1 if self.T <= 1 else 1 << (self.T - 1).bit_length()
        max_level = int(round(math.log2(T_pow2)))

        scores = np.zeros(self.m, dtype=np.float64)
        for i in range(self.m):
            t_star = int(self.leaf_of[i])
            for level in range(max_level + 1):
                idx = t_star >> level
                size = 1 << level
                a = idx * size
                b = min(a + size - 1, leaves_done - 1)
                if a > b:
                    continue
                clean_part = cum[b + 1, i] - cum[a, i]
                node_noise = self.node_noise_proj.get((level, idx), None)
                noise_part = float(node_noise[i]) if node_noise is not None else 0.0
                scores[i] += clean_part + noise_part
        return scores


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_dpftrl_whitebox(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    *,
    aug_multiplicity: int,
    max_physical_batch_size: int,
    logical_batch_size: int,
    target_steps: int,
    lr: float,
    max_grad_norm: float,
    sigma_node: float,
    canary_dirac_indices: List[Tuple[int, int]],
    canary_leaf_assignment: np.ndarray,
    canary_inclusion_mask: np.ndarray,
    canary_scale: float,
    ema_model: Optional[nn.Module] = None,
    ema_decay: float = 0.9999,
    ema_step_offset: int = 0,
    test_every: int = 20,
    test_fn=None,
    logger=None,
) -> Tuple[List[float], DPFTRLState, int]:
    """Run DP-FTRL until either the data loader is exhausted or `target_steps`
    leaves have been processed. Returns (per-step losses, state, leaves_done).
    """
    from opacus.grad_sample import GradSampleModule

    model = GradSampleModule(model)
    params = [p for p in model.parameters() if p.requires_grad]

    # Snapshot initial weights for FTRL update form theta_t = theta_0 - lr * G_t.
    theta_init = [p.detach().clone() for p in params]

    state = DPFTRLState(
        params=params,
        device=device,
        sigma_node=sigma_node,
        target_steps=target_steps,
        canary_dirac_indices=canary_dirac_indices,
        canary_leaf_assignment=canary_leaf_assignment,
        canary_inclusion_mask=canary_inclusion_mask,
        canary_scale=canary_scale,
    )

    criterion = nn.CrossEntropyLoss(reduction="none")
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    losses: List[float] = []
    samples_in_batch = 0
    accum_grad: List[torch.Tensor] = [torch.zeros_like(p) for p in params]
    leaves_done = 0

    pbar = tqdm(total=target_steps, desc="DP-FTRL", unit="leaf")
    while leaves_done < target_steps:
        for images, labels in train_loader:
            if leaves_done >= target_steps:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Augmentation multiplicity: B -> B*K with per-image augmentations.
            aug = [_augment_per_image(images) for _ in range(aug_multiplicity)]
            aug = torch.stack(aug).transpose(0, 1).reshape(-1, 3, 32, 32)
            aug = normalize(aug)
            aug_labels = labels.repeat_interleave(aug_multiplicity)

            # Forward + backward (per-sample grads via GradSampleModule).
            for p in params:
                if hasattr(p, "grad_sample"):
                    p.grad_sample = None
                p.grad = None
            outputs = model(aug)
            loss_per_sample = criterion(outputs, aug_labels)
            loss_per_sample.sum().backward()

            # Reduce B*K -> B by averaging over K augmentations.
            for p in params:
                gs = p.grad_sample
                B_K = gs.shape[0]
                B = B_K // aug_multiplicity
                p.grad_sample = gs.view(B, aug_multiplicity, *gs.shape[1:]).mean(dim=1)

            # Per-sample clip + sum (no noise here -- tree noise is added in state.step).
            clipped_sum = _clip_and_sum_grad_samples(params, max_grad_norm=max_grad_norm)
            for a, c in zip(accum_grad, clipped_sum):
                a.add_(c)
            samples_in_batch += images.shape[0]

            if samples_in_batch >= logical_batch_size:
                # ---- one DP-FTRL step ----
                noisy_G = state.step(accum_grad, leaves_done)
                step_scale = lr / float(logical_batch_size)
                with torch.no_grad():
                    for p, init, n in zip(params, theta_init, noisy_G):
                        p.data.copy_(init - step_scale * n)

                losses.append(float(loss_per_sample.mean().item()) * aug_multiplicity)
                leaves_done += 1
                samples_in_batch = 0
                for a in accum_grad:
                    a.zero_()
                pbar.update(1)

                # EMA on model parameters (TIMM-style warmup -- matches DP-SGD path).
                if ema_model is not None:
                    global_step = ema_step_offset + leaves_done
                    effective_decay = min(ema_decay, (1.0 + global_step) / (10.0 + global_step))
                    with torch.no_grad():
                        for p_ema, p in zip(ema_model.parameters(), params):
                            p_ema.data.mul_(effective_decay).add_(p.data, alpha=1.0 - effective_decay)
                        for b_ema, b in zip(ema_model.buffers(), model.buffers()):
                            b_ema.data.copy_(b.data)

                if leaves_done % 10 == 0:
                    pbar.set_postfix(step=leaves_done, loss=np.mean(losses[-10:]))

    return losses, state, leaves_done


def test(model: nn.Module, test_loader, device: torch.device) -> float:
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
    return 100.0 * correct / total
