import math
from typing import Union
from scipy.optimize import brentq
import numpy as np
Number = Union[float, int]

def _phi(x: float) -> float:
    """Standard normal CDF Φ(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def ndis_delta_1d_sigma1_lt_sigma2(sigma1, sigma2, mu1, mu2, eps, tol = 1e-15):
    r"""
        Compute delta, given eps, for X ~ N(μ1, σ1^2), Y ~ N(μ2, σ2^2), with σ1^2 < σ2^2.
    """
    sigma1 = float(sigma1)
    sigma2 = float(sigma2)
    mu1 = float(mu1)
    mu2 = float(mu2)
    eps = float(eps)

    if not (sigma1 > 0.0 and sigma2 > 0.0):
        raise ValueError("sigma1 and sigma2 must be positive.")
    if not (eps >= 0.0):
        raise ValueError("eps must be >= 0.")
    if not (sigma1 * sigma1 < sigma2 * sigma2):
        raise ValueError("Lemma requires sigma1^2 < sigma2^2.")

    tau = (sigma1 * sigma1) / (sigma2 * sigma2)
    a = 1.0 - tau
    if a <= tol:
        raise ValueError("sigma1^2 is too close to sigma2^2; a = 1 - tau is numerically unstable.")

    dmu = mu1 - mu2
    b = -sigma1 * dmu / (sigma2 * sigma2)
    c = eps + 0.5 * math.log(tau) - (dmu * dmu) / (2.0 * sigma2 * sigma2)

    D = b * b - 2.0 * a * c
    if D <= tol:   
        return 0.0

    sqrtD = math.sqrt(D)
    z_minus = (-b - sqrtD) / a
    z_plus  = (-b + sqrtD) / a
    if z_plus < z_minus:  # just in case
        z_minus, z_plus = z_plus, z_minus

    m = -dmu / sigma1
    sqrt_tau = math.sqrt(tau)

    term1 = _phi(z_plus) - _phi(z_minus)
    term2 = _phi((z_plus - m) * sqrt_tau) - _phi((z_minus - m) * sqrt_tau)
    delta = term1 - math.exp(eps) * term2

    return max(0.0, min(1.0, delta))



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
    """
    Return the smallest eps >= 0 such that delta(eps) <= delta_target
    using SciPy's brentq on f(eps) = delta(eps) - delta_target.

    Requires: pip install scipy
    """
    

    delta_target = float(delta_target)
    if not (0.0 <= delta_target <= 1.0):
        raise ValueError("delta_target must be in [0, 1].")

    def delta_of(eps: float) -> float:
        return ndis_delta_1d_sigma1_lt_sigma2(
            sigma1, sigma2, mu1, mu2, eps,
            tol=inner_tol
        )

    d0 = delta_of(0.0)
    if delta_target >= d0 - delta_tol:
        return 0.0

    # Find an upper bracket where delta(eps_max) <= target
    dmax = delta_of(eps_max)
    if dmax > delta_target + delta_tol:
        raise ValueError(
            f"eps_max={eps_max} too small: delta(eps_max)={dmax} still > target {delta_target}. "
            "Increase eps_max."
        )

    # Root for f(eps)=0 with f(0)>0 and f(eps_max)<0; monotone => smallest eps at root.
    def f(eps: float) -> float:
        return delta_of(eps) - delta_target

    # brentq returns a root in [0, eps_max]
    eps_star = brentq(f, 0.0, eps_max, xtol=xtol, rtol=rtol, maxiter=200)

    # If you want to be extra safe about "smallest eps", nudge left a hair and re-check.
    # Usually unnecessary, but cheap:
    eps_star = max(0.0, eps_star)
    return eps_star


def estimate_mean_variance(in_scores, out_scores):
    """
    Estimate mean and variance for both in_scores and out_scores.
    
    Parameters:
    -----------
    in_scores : numpy.ndarray
        Array of scores for in-canaries
    out_scores : numpy.ndarray
        Array of scores for out-canaries
    
    Returns:
    --------
    dict : Dictionary containing:
        - in_mean : mean of in_scores
        - in_var : variance of in_scores
        - in_std : standard deviation of in_scores
        - out_mean : mean of out_scores
        - out_var : variance of out_scores
        - out_std : standard deviation of out_scores
    """
    in_mean = np.mean(in_scores)
    in_var = np.var(in_scores, ddof=0)  # Population variance
    in_std = np.std(in_scores, ddof=0)   # Population standard deviation
    
    out_mean = np.mean(out_scores)
    out_var = np.var(out_scores, ddof=0)  # Population variance
    out_std = np.std(out_scores, ddof=0)   # Population standard deviation
    
    return {
        'in_mean': in_mean,
        'in_var': in_var,
        'in_std': in_std,
        'out_mean': out_mean,
        'out_var': out_var,
        'out_std': out_std
    }