"""Clean simulator for a T-fold composition of (sub-sampled) Laplace mechanisms.

Mirrors the Gaussian DP-SGD-style simulator in
``src/classifier/white_box_dp_sgd.py``: produces per-canary observation
sequences in {out, in} that can be aggregated into 1D scores and fed to
either ``CanaryScoreAuditor`` (auditing.py) or the 1D NDIS auditor
(``whitebox_auditing/ndis_1d.py``).

Per-step model (sensitivity = ``mu``, scale = ``b``):
    H0 (out canary):   X_t ~ Laplace(0, b)
    H1 (in  canary):   X_t ~ (1 - q) * Laplace(0, b) + q * Laplace(mu, b)

Special cases:
    q == 1  -> pure Laplace composition (e.g. tree mechanism, linear
               contextual bandits): every step Lap(0,b) vs Lap(mu,b).
    q  < 1  -> sub-sampled Laplace (analogue of sub-sampled Gaussian /
               DP-SGD); under H1 the per-step variance is strictly larger
               than under H0, so the NDIS Gaussian-vs-Gaussian approximation
               is non-degenerate.

Score conventions match the existing Gaussian simulator: the per-canary
1D score is the sum across the T steps. Downstream code may divide by T
(time-average) and rescale by sqrt(T) for the CLT-Gaussian approximation
used by NDIS.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def sample_laplace(T: int, n: int, b: float, rng: np.random.Generator) -> np.ndarray:
    """Out-canary observations: i.i.d. Lap(0, b), shape (n, T)."""
    return rng.laplace(loc=0.0, scale=float(b), size=(n, T))


def sample_laplace_mixture(
    T: int,
    n: int,
    q: float,
    mu: float,
    b: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """In-canary observations under sub-sampled Laplace.

    With probability ``q`` per step the canary's contribution is included
    (location ``mu``); otherwise the location is 0.

    Returns
    -------
    samples : np.ndarray of shape (n, T)
    indicators : np.ndarray of shape (n, T), bool
        True where the in-canary was actually sampled in that step.
    """
    if not (0.0 < q <= 1.0):
        raise ValueError(f"q must be in (0, 1], got {q}.")

    indicators = rng.random((n, T)) < q
    means = np.where(indicators, mu, 0.0)
    samples = rng.laplace(loc=means, scale=float(b), size=(n, T))
    return samples, indicators


def per_step_eps_basic(b: float, mu: float, q: float = 1.0) -> float:
    """Per-step pure-DP epsilon.

    Pure Laplace (q=1):       eps_step = mu / b.
    Sub-sampled Laplace (q<1): eps_step = log(1 + q * (exp(mu/b) - 1))
        (privacy amplification by sub-sampling for pure-DP).
    """
    if not (0.0 < q <= 1.0):
        raise ValueError(f"q must be in (0, 1], got {q}.")
    eps0 = float(mu) / float(b)
    if q >= 1.0:
        return eps0
    return math.log1p(q * math.expm1(eps0))


def eps_basic_composition(
    T: int, b: float, mu: float = 1.0, q: float = 1.0
) -> float:
    """Basic (pure-DP) composition upper bound after T steps."""
    return T * per_step_eps_basic(b=b, mu=mu, q=q)


def eps_advanced_composition(
    T: int, b: float, delta: float, mu: float = 1.0, q: float = 1.0
) -> float:
    """Advanced-composition (Dwork et al.) upper bound after T steps.

    Returns min(basic, advanced) since basic can be tighter when T is small
    or eps_step is small.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}.")
    eps0 = per_step_eps_basic(b=b, mu=mu, q=q)
    eps_basic = T * eps0
    eps_adv = (
        math.sqrt(2.0 * T * math.log(1.0 / delta)) * eps0
        + T * eps0 * math.expm1(eps0)
    )
    return min(eps_basic, eps_adv)


def ndis_gaussian_params(
    T: int, b: float, mu: float = 1.0, q: float = 1.0
) -> dict:
    """CLT Gaussian-approximation parameters for the sum-of-T score.

    Per step: Var(Lap(0,b)) = 2 b^2.
    Under H0 (out): score_T = sum_t X_t  =>  mean=0, var = T * 2 b^2.
    Under H1 (in):  per-step mean = q*mu, per-step var = 2 b^2 + q(1-q) mu^2
        =>  score_T mean = T*q*mu, var = T*(2 b^2 + q(1-q) mu^2).

    Following the convention used in ``simulation_one_run_auditing.ipynb``
    we work with the rescaled score s = sum / sqrt(T), which has the
    *per-step* variance and a sqrt(T)-amplified mean shift. This gives
    sigma1 < sigma2 whenever q < 1 (so the NDIS lemma applies); for q = 1
    sigma1 == sigma2 and the user should fall back to a pure mean-shift
    Gaussian DP calculation (or to ``CanaryScoreAuditor.epsilon_one_run``).
    """
    var_step_out = 2.0 * b * b
    var_step_in = 2.0 * b * b + q * (1.0 - q) * mu * mu
    return {
        "mu1": 0.0,
        "sigma1": math.sqrt(var_step_out),
        "mu2": math.sqrt(T) * q * mu,
        "sigma2": math.sqrt(var_step_in),
    }