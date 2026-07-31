"""Quantile-based adaptive clipping (Andrew et al. 2021, arXiv:1905.03871).

Opacus ships the mechanism as AdaClipDPOptimizer; this adds a zero_grad fix and
the hyperparameter block. Scores are normalized by C_t in gen_scores_DP_whitebox.py.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from opacus.optimizers import AdaClipDPOptimizer
from opacus.optimizers.optimizer import DPOptimizer


class AdaClipDPOptimizerFixed(AdaClipDPOptimizer):
    """AdaClipDPOptimizer whose quantile counters survive gradient accumulation.

    Stock zero_grad resets sample_size/unclipped_num without the parent's
    _is_last_step_skipped guard, so BatchMemoryManager wipes them on every
    physical sub-batch: 128 samples instead of 4096, and sigma_b/128 instead of
    sigma_b/4096 noise on the unclipped fraction.
    """

    def zero_grad(self, set_to_none: bool = False):
        skipped = self._is_last_step_skipped
        DPOptimizer.zero_grad(self, set_to_none)  # skip AdaClip's unguarded reset
        if not skipped:
            self.sample_size = 0
            self.unclipped_num = 0


@dataclass
class AdaptiveClipConfig:
    """gamma and eta_C are the paper's values; the clamps are ours (Opacus needs
    bounds, Algorithm 1 has none). C_0 is the optimizer's initial max_grad_norm."""

    target_unclipped_quantile: float = 0.5    # clip to the median
    clipbound_learning_rate: float = 0.2      # paper Sec. 2.1
    unclipped_num_std: float = 204.8          # sigma_b; paper rule is m/20
    min_clipbound: float = 0.1
    max_clipbound: float = 50.0               # loose: a binding clamp = fixed clipping

    @classmethod
    def for_batch_size(cls, expected_batch_size: int, **overrides) -> "AdaptiveClipConfig":
        kwargs = {"unclipped_num_std": expected_batch_size / 20.0}
        kwargs.update(overrides)
        return cls(**kwargs)

    def as_engine_kwargs(self) -> dict:
        return asdict(self)

    def validate(self, noise_multiplier: float) -> None:
        if self.max_clipbound <= self.min_clipbound:
            raise ValueError("max_clipbound must be larger than min_clipbound.")
        if noise_multiplier >= 2 * self.unclipped_num_std:
            raise ValueError(f"noise_multiplier={noise_multiplier:.4f} must be < 2*unclipped_num_std={2 * self.unclipped_num_std:.4f} (Theorem 1). Raise --unclipped-num-std.")


def install_adaptive_clipping_fix(optimizer) -> None:
    """Swap in the fixed class; Opacus picks the optimizer class with no injection point."""
    if not isinstance(optimizer, AdaClipDPOptimizer):
        raise TypeError(f"expected AdaClipDPOptimizer, got {type(optimizer).__name__}")
    optimizer.__class__ = AdaClipDPOptimizerFixed


def effective_noise_multipliers(target_noise_multiplier: float, unclipped_num_std: float):
    """(sigma_eff, sigma_grad) under the Theorem 1 split. Opacus overwrites
    noise_multiplier with sigma_grad, which the accountant then reads without ever
    charging for the count release, so get_epsilon() reads low."""
    sigma_g = (target_noise_multiplier ** (-2) - (2 * unclipped_num_std) ** (-2)) ** (-0.5)
    return target_noise_multiplier, sigma_g
