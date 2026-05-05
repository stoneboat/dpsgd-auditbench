"""Clean simulator for the binary-tree (a.k.a. Honaker / DPCP) Gaussian mechanism over a stream of T scalar leaves."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def num_levels(T: int) -> int:
    """Number of internal levels above the leaves (== ceil(log2 T))."""
    return int(math.ceil(math.log2(_next_pow2(T))))


def prefix_decomposition(t: int) -> List[Tuple[int, int]]:
    """Canonical set of tree nodes covering the prefix [0..t] (inclusive).
    """
    if t < 0:
        return []
    nodes: List[Tuple[int, int]] = []
    n = t + 1
    offset = 0
    msb = n.bit_length() - 1
    for level in range(msb, -1, -1):
        if n & (1 << level):
            nodes.append((level, offset >> level))
            offset += 1 << level
    return nodes


def tree_release(leaves: np.ndarray, sigma: float, rng: np.random.Generator) -> Dict[Tuple[int, int], float]:
    """Build the noised binary tree over `leaves`.
    """
    T = _next_pow2(len(leaves))
    padded = np.zeros(T, dtype=float)
    padded[: len(leaves)] = leaves

    tree: Dict[Tuple[int, int], float] = {}
    # Level 0: leaves with independent noise.
    noise0 = rng.normal(loc=0.0, scale=sigma, size=T)
    for i in range(T):
        tree[(0, i)] = float(padded[i] + noise0[i])
    clean = padded.copy()
    L = num_levels(T)
    for level in range(1, L + 1):
        size = 1 << level
        n_nodes = T // size
        clean = clean.reshape(n_nodes, 2).sum(axis=1)
        noise = rng.normal(loc=0.0, scale=sigma, size=n_nodes)
        for idx in range(n_nodes):
            tree[(level, idx)] = float(clean[idx] + noise[idx])

    return tree


def prefix_sum_at(tree: Dict[Tuple[int, int], float], t: int) -> float:
    """Noisy estimate of sum(leaves[:t+1]) using the canonical O(log t) nodes."""
    return float(sum(tree[node] for node in prefix_decomposition(t)))


# ---------------------------------------------------------------------------
# Vectorized simulators (no full tree construction needed for the audit).
# ---------------------------------------------------------------------------
def num_covering_nodes(t: int) -> int:
    """popcount(t+1): number of tree nodes in the canonical prefix [0..t]."""
    return bin(t + 1).count("1")


def ancestors(t: int, T: int) -> List[Tuple[int, int]]:
    """All log_2(T) + 1 tree nodes on the leaf-to-root path from leaf `t`.
    """
    L = num_levels(_next_pow2(T))
    return [(level, t >> level) for level in range(L + 1)]


def sample_optimal_audit_scores(t_star: int, sigma: float, n: int, rng: np.random.Generator, *, T: int, canary_mu: float = 0.0,) -> np.ndarray:
    """SNR-optimal audit statistic: sum of noisy values over ancestors of `t_star`.

    The canary contributes `canary_mu` to leaf `t_star`, and every one of
    the `L = log_2(T) + 1` ancestor nodes inherits that contribution. With
    independent Gaussian noise per node, summing the L ancestor values
    gives mean `L * canary_mu` and variance `L * sigma^2`. The standardized
    SNR is `sqrt(L) * canary_mu / sigma` -- exactly the GDP parameter of
    the underlying tree mechanism, with no leftover sqrt(log T) factor.
    """
    L = num_levels(_next_pow2(T)) + 1   # leaf + log2(T) internal ancestors
    return canary_mu * L + rng.normal(0.0, sigma * math.sqrt(L), size=n)


def sample_prefix_scores(t_star: int, sigma: float, n: int, rng: np.random.Generator, *, canary_mu: float = 0.0, leaf_baseline: float = 0.0,
) -> np.ndarray:
    """Simulate the noisy prefix sum at time t_star for `n` independent runs.

    Parameters
    ----------
    t_star : int
        Leaf index where the canary (if any) contributes.
    sigma : float
        Per-node Gaussian noise standard deviation.
    canary_mu : float
        Canary contribution to leaf t_star. Use 0.0 for "out", mu for "in".
    leaf_baseline : float
        Constant baseline value at every clean leaf (typically 0 in audit
        simulations; non-zero is supported for sanity checks).
    """
    k = num_covering_nodes(t_star)
    # True prefix sum (noiseless, deterministic) at t_star:
    true_prefix = leaf_baseline * (t_star + 1) + canary_mu
    # Aggregate noise across the k covering nodes is N(0, k * sigma^2).
    noise = rng.normal(loc=0.0, scale=sigma * math.sqrt(k), size=n)
    return true_prefix + noise


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
    # Phi(-a + mu/2) - exp(eps) * Phi(-a - mu/2)
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

# ---------------------------------------------------------------------------
# NDIS Gaussian-vs-Gaussian parameters at one prefix observation point.
# ---------------------------------------------------------------------------
def ndis_gaussian_params(t_star: int, sigma: float, mu: float = 1.0) -> dict:
    k = num_covering_nodes(t_star)
    s = sigma * math.sqrt(k)
    return {
        "mu1": 0.0,
        "sigma1": s,
        "mu2": float(mu),
        "sigma2": s,
        "k": k,
    }