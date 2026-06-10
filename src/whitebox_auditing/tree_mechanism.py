"""Privacy calibration helpers for the binary-tree (Honaker / DPCP) Gaussian
mechanism over a stream of T scalar leaves.

Only the (eps, delta) <-> sigma_node calibration is needed at training time;
the full simulator (clean tree construction, prefix sums, optimal audit
score sampling) was removed alongside the synthetic-audit notebook.
"""

from __future__ import annotations

import math


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def num_levels(T: int) -> int:
    """Number of internal levels above the leaves (== ceil(log2 T))."""
    return int(math.ceil(math.log2(_next_pow2(T))))


# ---------------------------------------------------------------------------
# Privacy accounting: per-node sigma calibrated to (eps, delta).
# ---------------------------------------------------------------------------
def _balle_wang_delta(eps: float, mu_gdp: float) -> float:
    """Tight (eps, delta) for the Gaussian mechanism with parameter
    mu_gdp = sensitivity / sigma. Balle & Wang (2018), Theorem 8.
    """
    if mu_gdp <= 0.0:
        return 0.0
    a = eps / mu_gdp
    from math import erf
    phi = lambda z: 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
    return phi(-a + 0.5 * mu_gdp) - math.exp(eps) * phi(-a - 0.5 * mu_gdp)


def tree_sigma_for_eps(target_eps: float, T: int, delta: float, *, sigma_max: float = 1e6) -> float:
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}.")
    if target_eps <= 0.0:
        raise ValueError(f"target_eps must be positive, got {target_eps}.")
    L = max(1, num_levels(T)) + 1   # +1 for the noisy leaf level
    sens = math.sqrt(L)

    lo, hi = 1e-6, sigma_max
    while _balle_wang_delta(target_eps, sens / hi) > delta:
        hi *= 2.0
        if hi > sigma_max * 1024:
            raise RuntimeError("Cannot achieve target (eps, delta).")
    for _ in range(80):
        mid = math.sqrt(lo * hi)  # geometric bisection
        if _balle_wang_delta(target_eps, sens / mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def tree_eps_for_sigma(sigma: float, T: int, delta: float, *, eps_max: float = 100.0) -> float:
    """Inverse: tightest eps for the tree mechanism at given per-node sigma."""
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    L = max(1, num_levels(T)) + 1   # +1 for the noisy leaf level
    sens = math.sqrt(L)
    mu_gdp = sens / sigma

    lo, hi = 0.0, eps_max
    while _balle_wang_delta(hi, mu_gdp) > delta:
        hi *= 2.0
        if hi > eps_max * 1024:
            raise RuntimeError("Could not bracket epsilon.")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _balle_wang_delta(mid, mu_gdp) > delta:
            lo = mid
        else:
            hi = mid
    return hi
