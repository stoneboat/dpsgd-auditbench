"""Matrix-factorization DP-FTRL: strategy/decoder matrices and calibration.

The workload is the prefix-sum matrix A (lower-triangular ones). We factor
A = B @ C, release C@x + nu with nu ~ N(0, sigma^2 I) i.i.d., and decode to
A@x + B@nu. The tree mechanism is the special case where C is the tree encoder.

C is the (optionally banded) matrix square root of A, which is lower-triangular
Toeplitz with coefficients from (1-x)^{-1/2}. Closed form, no optimization to
solve (Kalinin & Lampert, arXiv:2405.13763).
"""

from __future__ import annotations

import math

import numpy as np

from .tree_mechanism import _balle_wang_delta


def prefix_sum_matrix(T: int) -> np.ndarray:
    return np.tril(np.ones((T, T), dtype=np.float64))


def sqrt_toeplitz_coeffs(T: int) -> np.ndarray:
    """Taylor coefficients of (1-x)^{-1/2}: c_k = c_{k-1} * (2k-1)/(2k)."""
    c = np.empty(T, dtype=np.float64)
    c[0] = 1.0
    for k in range(1, T):
        c[k] = c[k - 1] * (2 * k - 1) / (2 * k)
    return c


def sqrt_strategy(T: int, bands: int = 0) -> np.ndarray:
    """Lower-triangular Toeplitz sqrt of the prefix-sum matrix.

    bands=0 keeps the full square root (exact, and B == C). bands=b zeroes
    coefficients at lag >= b, which bounds memory and multi-epoch sensitivity.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")
    if bands < 0 or bands > T:
        raise ValueError(f"bands must be in [0, T], got {bands}.")
    c = sqrt_toeplitz_coeffs(T)
    if bands:
        c[bands:] = 0.0
    lags = np.arange(T)[:, None] - np.arange(T)[None, :]
    return np.where(lags >= 0, c[np.clip(lags, 0, T - 1)], 0.0)


def decoder(C: np.ndarray) -> np.ndarray:
    """B = A @ inv(C), solved as C.T @ B.T = A.T."""
    A = prefix_sum_matrix(C.shape[0])
    return np.linalg.solve(C.T, A.T).T


def column_norms(C: np.ndarray) -> np.ndarray:
    return np.linalg.norm(C, axis=0)


def sensitivity(C: np.ndarray, clip_norm: float = 1.0) -> float:
    """L2 sensitivity of x -> C@x under single participation."""
    return float(clip_norm * column_norms(C).max())


def sigma_for_eps(target_eps: float, sens: float, delta: float, *, sigma_max: float = 1e6) -> float:
    """Gaussian sigma achieving (eps, delta) at the given L2 sensitivity."""
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}.")
    if target_eps <= 0.0:
        raise ValueError(f"target_eps must be positive, got {target_eps}.")
    lo, hi = 1e-6, sigma_max
    while _balle_wang_delta(target_eps, sens / hi) > delta:
        hi *= 2.0
        if hi > sigma_max * 1024:
            raise RuntimeError("Cannot achieve target (eps, delta).")
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if _balle_wang_delta(target_eps, sens / mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def eps_for_sigma(sigma: float, sens: float, delta: float, *, eps_max: float = 100.0) -> float:
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
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


def audit_gaussian_params(C: np.ndarray, sigma: float, leaf: int, clip_norm: float = 1.0):
    """Analytic (mu_out, mu_in, sd) of the normalized score for a canary at `leaf`.

    Score is <C[:, leaf], C@x + nu> / ||C[:, leaf]||, so both worlds have sd sigma
    and the in-world mean is clip_norm * ||C[:, leaf]||. Exactly Gaussian: no CLT.
    """
    col = float(np.linalg.norm(C[:, leaf]))
    return 0.0, clip_norm * col, sigma
