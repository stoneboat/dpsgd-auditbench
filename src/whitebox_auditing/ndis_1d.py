"""Normal-Distribution Indistinguishability (NDIS) audit primitives.
"""
import math
from typing import Optional, Union

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

Number = Union[float, int]

_norm_logcdf = norm.logcdf
_NEG_INF = -np.inf


def _log_delta_two_gaussians_one_direction(mu1: float, std1: float, mu2: float, std2: float, eps: float, *, equal_var_tol: float = 1e-12,) -> float:
    a = 0.5 * (std2**-2 - std1**-2)
    b = mu1 / (std1**2) - mu2 / (std2**2)
    c = (0.5 * ((mu2 / std2) ** 2 - (mu1 / std1) ** 2) + math.log(std2) - math.log(std1))

    if abs(a) < equal_var_tol:
        # Equal-variance limit. Linear: {L > eps} = {b z + c > eps}.
        if abs(b) < equal_var_tol:
            # No mean shift either -> distributions identical -> delta = 0.
            return _NEG_INF if c <= eps else 0.0
        threshold = (eps - c) / b
        if b > 0:
            # {z > threshold}; Pr_W[Z > t] = Phi((mu - t)/std).
            pr_z1 = _norm_logcdf(mu1, loc=threshold, scale=std1)
            pr_z2 = _norm_logcdf(mu2, loc=threshold, scale=std2)
        else:
            # {z < threshold}; Pr_W[Z < t] = Phi((t - mu)/std).
            pr_z1 = _norm_logcdf(threshold, loc=mu1, scale=std1)
            pr_z2 = _norm_logcdf(threshold, loc=mu2, scale=std2)
    else:
        determinant = b * b - 4.0 * a * (c - eps)
        if determinant <= 0:
            return _NEG_INF

        sqrtD = math.sqrt(determinant)
        i_left = (-b - sqrtD) / (2.0 * a)
        i_right = (-b + sqrtD) / (2.0 * a)
        if i_left > i_right:
            i_left, i_right = i_right, i_left

        if a > 0:
            pr_z1 = np.logaddexp(_norm_logcdf(i_left, loc=mu1, scale=std1),_norm_logcdf(mu1, loc=i_right, scale=std1))
            pr_z2 = np.logaddexp(_norm_logcdf(i_left, loc=mu2, scale=std2),_norm_logcdf(mu2, loc=i_right, scale=std2))
        else:
            pr_z1 = logsumexp([_norm_logcdf(i_right, loc=mu1, scale=std1), _norm_logcdf(i_left, loc=mu1, scale=std1)], b=[1.0, -1.0])
            pr_z2 = logsumexp([_norm_logcdf(i_right, loc=mu2, scale=std2), _norm_logcdf(i_left, loc=mu2, scale=std2)],b=[1.0, -1.0])

    if pr_z1 == _NEG_INF:
        return _NEG_INF
    if pr_z2 == _NEG_INF:
        return pr_z1
    factor = eps + pr_z2 - pr_z1
    if factor >= 0.0:
        return _NEG_INF
    return pr_z1 + math.log1p(-math.exp(factor))


def log_delta_two_gaussians(
    mu1: float, std1: float, mu2: float, std2: float, eps: float,
) -> float:
    log_d_xy = _log_delta_two_gaussians_one_direction(mu1, std1, mu2, std2, eps)
    log_d_yx = _log_delta_two_gaussians_one_direction(mu2, std2, mu1, std1, eps)
    return max(log_d_xy, log_d_yx)


def ndis_delta_1d_sigma1_lt_sigma2(
    sigma1: Number, sigma2: Number, mu1: Number, mu2: Number, eps: Number,
    tol: float = 1e-15,
) -> float:
    if not (sigma1 > 0.0 and sigma2 > 0.0):
        raise ValueError("sigma1 and sigma2 must be positive.")
    if not (eps >= 0.0):
        raise ValueError("eps must be >= 0.")
    log_d = _log_delta_two_gaussians_one_direction(float(mu1), float(sigma1), float(mu2), float(sigma2), float(eps), equal_var_tol=tol)
    if log_d == _NEG_INF:
        return 0.0
    return float(min(1.0, math.exp(log_d)))


def ndis_eps_from_delta_1d_brentq(
    sigma1: Number,
    sigma2: Number,
    mu1: Number,
    mu2: Number,
    delta_target: Number,
    *,
    eps_max: float = 100.0,
    xtol: float = 1e-10,
    rtol: float = 1e-10,
    delta_tol: float = 1e-12,
    inner_tol: float = 1e-15,
) -> float:
    """Smallest eps >= 0 with delta(eps) <= delta_target.

    Uses the symmetric (max-over-directions) hockey-stick, so the result is
    a tight DP eps regardless of which Gaussian is passed as 1 vs 2 and
    regardless of whether the variances are equal.
    """
    delta_target = float(delta_target)
    if not (0.0 <= delta_target <= 1.0):
        raise ValueError("delta_target must be in [0, 1].")
    if not (sigma1 > 0.0 and sigma2 > 0.0):
        raise ValueError("sigma1 and sigma2 must be positive.")

    log_dt = math.log(delta_target) if delta_target > 0.0 else _NEG_INF

    def log_delta_of(eps: float) -> float:
        return log_delta_two_gaussians(float(mu1), float(sigma1), float(mu2), float(sigma2), float(eps))

    # Already (0, delta)-DP?
    if log_delta_of(0.0) <= log_dt + delta_tol:
        return 0.0

    # Bracket: find eps_hi with log_delta(eps_hi) <= log_dt.
    eps_hi = max(1.0, eps_max * 0.0 + 1.0)
    while log_delta_of(eps_hi) > log_dt:
        eps_hi *= 2.0
        if eps_hi > eps_max:
            if log_delta_of(eps_max) > log_dt:
                raise ValueError(
                    f"eps_max={eps_max} too small: log_delta(eps_max)="
                    f"{log_delta_of(eps_max)} still > log delta_target={log_dt}."
                )
            eps_hi = eps_max
            break

    def f(eps: float) -> float:
        # Monotone in log space; brentq on log_delta - log_dt is well-conditioned.
        return log_delta_of(eps) - log_dt

    eps_star = brentq(f, 0.0, eps_hi, xtol=xtol, rtol=rtol, maxiter=200)
    return max(0.0, eps_star)


def estimate_mean_variance(in_scores, out_scores) -> dict:
    """Sample moments of the in/out canary score distributions.

    Uses the unbiased (ddof=1) estimator for variance/std. The biased
    population estimator (ddof=0) underestimates sigma and therefore
    overestimates the GDP parameter mu/sigma, biasing the audit eps up.
    """
    in_scores = np.asarray(in_scores)
    out_scores = np.asarray(out_scores)
    return {
        'in_mean': float(np.mean(in_scores)),
        'in_var': float(np.var(in_scores, ddof=1)),
        'in_std': float(np.std(in_scores, ddof=1)),
        'out_mean': float(np.mean(out_scores)),
        'out_var': float(np.var(out_scores, ddof=1)),
        'out_std': float(np.std(out_scores, ddof=1)),
    }


def _ndis_eps_from_moments(in_mean: float, in_std: float, out_mean: float, out_std: float, delta: float, *, pool_variance: bool = False) -> float:
    if pool_variance:
        s = math.sqrt(0.5 * (in_std * in_std + out_std * out_std))
        in_std = out_std = s
    try:
        return ndis_eps_from_delta_1d_brentq(sigma1=out_std, sigma2=in_std,mu1=out_mean, mu2=in_mean,delta_target=delta)
    except (ValueError, RuntimeError):
        return 0.0


def ndis_eps_lower_bound_with_ci(in_scores, out_scores, delta: float, *, alpha: float = 0.05, n_bootstrap: int = 2000, pool_variance: bool = True, rng: Optional[np.random.Generator] = None, return_samples: bool = False):
    """Bootstrap (1-alpha)-confidence lower bound on the NDIS eps.
    Resamples (in_scores, out_scores) with replacement, computes the NDIS
    """
    if rng is None:
        rng = np.random.default_rng()
    in_scores = np.asarray(in_scores, dtype=float)
    out_scores = np.asarray(out_scores, dtype=float)
    n_in, n_out = len(in_scores), len(out_scores)
    if n_in < 2 or n_out < 2:
        raise ValueError(f"Need >=2 samples in each group; got n_in={n_in}, n_out={n_out}.")

    eps_samples = np.empty(n_bootstrap, dtype=float)
    for k in range(n_bootstrap):
        in_b = rng.choice(in_scores, size=n_in, replace=True)
        out_b = rng.choice(out_scores, size=n_out, replace=True)
        eps_samples[k] = _ndis_eps_from_moments(
            in_mean=float(np.mean(in_b)),
            in_std=float(np.std(in_b, ddof=1)),
            out_mean=float(np.mean(out_b)),
            out_std=float(np.std(out_b, ddof=1)),
            delta=delta,
            pool_variance=pool_variance,
        )

    eps_lb = float(np.quantile(eps_samples, alpha))
    if return_samples:
        return eps_lb, eps_samples
    return eps_lb
