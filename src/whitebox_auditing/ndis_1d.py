"""Normal-Distribution Indistinguishability (NDIS) audit primitives.
"""
import math
from typing import Optional, Tuple, Union

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp
from scipy.stats import chi2, norm, t as student_t

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


# ---------------------------------------------------------------------------
# Confidence-region -> worst-case-eps lower bounds
# ---------------------------------------------------------------------------
#
# Notation. Let theta = (mu_in, sigma_in, mu_out, sigma_out). The NDIS
# functional eps(theta; delta) returns the smallest eps such that the
# symmetric hockey-stick divergence between N(mu_in, sigma_in^2) and
# N(mu_out, sigma_out^2) at level delta is non-positive. This is what
# ndis_eps_from_delta_1d_brentq computes.
#
# A (1 - alpha)-confidence region C_alpha for theta gives a (1 - alpha)
# lower bound on the realised eps via
#       eps_LB := inf_{theta in C_alpha} eps(theta; delta).
# We provide several constructions of C_alpha.


def _eps_of_theta(theta: np.ndarray, delta: float, pool_variance: bool = False) -> float:
    mu_in, sigma_in, mu_out, sigma_out = theta
    if sigma_in <= 0.0 or sigma_out <= 0.0:
        return 0.0
    return _ndis_eps_from_moments(
        in_mean=mu_in, in_std=sigma_in,
        out_mean=mu_out, out_std=sigma_out,
        delta=delta, pool_variance=pool_variance,
    )


def _minimize_eps_over_box(
    lo: np.ndarray, hi: np.ndarray, delta: float,
    pool_variance: bool = False, n_starts: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, np.ndarray]:
    """Minimise eps(theta; delta) over the rectangle [lo, hi].

    Combines (a) evaluation at every corner (16 of them in 4-D), which is
    the global minimiser whenever eps is monotone in each coord, and
    (b) a few L-BFGS-B restarts to catch interior minima if monotonicity
    fails. Returns (eps_min, theta_min).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    lo = np.maximum(lo, [-(np.inf), 1e-12, -(np.inf), 1e-12])

    # Corner search.
    best_eps = np.inf
    best_theta = None
    for mask in range(16):
        corner = np.where(np.array([(mask >> b) & 1 for b in range(4)], dtype=bool), hi, lo)
        e = _eps_of_theta(corner, delta, pool_variance=pool_variance)
        if e < best_eps:
            best_eps, best_theta = e, corner

    # Numeric refinement; cheap insurance against non-monotone minima.
    bounds = list(zip(lo.tolist(), hi.tolist()))

    def obj(theta):
        return _eps_of_theta(theta, delta, pool_variance=pool_variance)

    starts = [best_theta]
    for _ in range(max(0, n_starts - 1)):
        starts.append(rng.uniform(lo, hi))
    for x0 in starts:
        try:
            res = minimize(obj, x0=x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 50, 'ftol': 1e-9})
            if res.fun < best_eps:
                best_eps = float(res.fun)
                best_theta = res.x
        except Exception:
            pass

    return max(0.0, float(best_eps)), best_theta


# ---- (A) Parametric (Gaussian-assumption) Bonferroni rectangle --------------

def ndis_eps_lb_parametric_bonferroni(
    in_scores, out_scores, delta: float, *,
    alpha: float = 0.05, pool_variance: bool = False,
    n_dim: int = 4,
):
    """Lower bound via Gaussian parametric CIs + Bonferroni rectangle.

    Assumes the in/out scores are Gaussian. Marginal CIs at level
    1 - alpha/n_dim:
      mu      ~ Student-t   CI on the sample mean
      sigma^2 ~ chi^2       CI on the sample variance
    The hyperrectangle of these four CIs has joint coverage >= 1 - alpha
    by Bonferroni. Returns the minimum NDIS eps over that rectangle.

    n_dim is the number of CIs Bonferroni-corrected over. Default 4 covers
    (mu_in, sigma_in, mu_out, sigma_out). If pool_variance=True you may
    set n_dim=3 to correct over (mu_in, mu_out, sigma_pooled).
    """
    in_scores = np.asarray(in_scores, dtype=float)
    out_scores = np.asarray(out_scores, dtype=float)
    n_in, n_out = len(in_scores), len(out_scores)
    if n_in < 2 or n_out < 2:
        raise ValueError("Need >= 2 samples in each group.")

    a_marg = alpha / n_dim
    mu_in_hat, mu_out_hat = float(np.mean(in_scores)), float(np.mean(out_scores))
    s2_in = float(np.var(in_scores, ddof=1))
    s2_out = float(np.var(out_scores, ddof=1))

    # Student-t CI on the mean.
    t_in = student_t.ppf(1.0 - a_marg / 2.0, df=n_in - 1)
    t_out = student_t.ppf(1.0 - a_marg / 2.0, df=n_out - 1)
    mu_in_lo = mu_in_hat - t_in * math.sqrt(s2_in / n_in)
    mu_in_hi = mu_in_hat + t_in * math.sqrt(s2_in / n_in)
    mu_out_lo = mu_out_hat - t_out * math.sqrt(s2_out / n_out)
    mu_out_hi = mu_out_hat + t_out * math.sqrt(s2_out / n_out)

    # Chi^2 CI on the variance: (n-1) S^2 / sigma^2 ~ chi^2_{n-1}.
    chi2_in_lo = chi2.ppf(a_marg / 2.0, df=n_in - 1)
    chi2_in_hi = chi2.ppf(1.0 - a_marg / 2.0, df=n_in - 1)
    chi2_out_lo = chi2.ppf(a_marg / 2.0, df=n_out - 1)
    chi2_out_hi = chi2.ppf(1.0 - a_marg / 2.0, df=n_out - 1)
    sig_in_lo = math.sqrt((n_in - 1) * s2_in / chi2_in_hi)
    sig_in_hi = math.sqrt((n_in - 1) * s2_in / chi2_in_lo)
    sig_out_lo = math.sqrt((n_out - 1) * s2_out / chi2_out_hi)
    sig_out_hi = math.sqrt((n_out - 1) * s2_out / chi2_out_lo)

    lo = np.array([mu_in_lo, sig_in_lo, mu_out_lo, sig_out_lo])
    hi = np.array([mu_in_hi, sig_in_hi, mu_out_hi, sig_out_hi])
    eps_lb, theta_min = _minimize_eps_over_box(lo, hi, delta, pool_variance=pool_variance)
    return {
        'eps_lb': eps_lb,
        'theta_min': theta_min,
        'box_lo': lo, 'box_hi': hi,
        'method': 'parametric_bonferroni',
    }


# ---- (B) Bootstrap-percentile Bonferroni rectangle --------------------------

def _bootstrap_param_samples(
    in_scores: np.ndarray, out_scores: np.ndarray,
    n_bootstrap: int, rng: np.random.Generator,
) -> np.ndarray:
    """Resamples each group with replacement, returns (B, 4) parameter draws."""
    n_in, n_out = len(in_scores), len(out_scores)
    out = np.empty((n_bootstrap, 4), dtype=float)
    for b in range(n_bootstrap):
        ib = rng.choice(in_scores, size=n_in, replace=True)
        ob = rng.choice(out_scores, size=n_out, replace=True)
        out[b, 0] = np.mean(ib)
        out[b, 1] = np.std(ib, ddof=1)
        out[b, 2] = np.mean(ob)
        out[b, 3] = np.std(ob, ddof=1)
    return out


def ndis_eps_lb_bootstrap_bonferroni(
    in_scores, out_scores, delta: float, *,
    alpha: float = 0.05, n_bootstrap: int = 2000,
    pool_variance: bool = False, n_dim: int = 4,
    rng: Optional[np.random.Generator] = None,
):
    """Lower bound via bootstrap percentile CIs + Bonferroni rectangle.

    Each marginal CI is the (alpha/(2 n_dim), 1 - alpha/(2 n_dim))
    bootstrap-percentile CI; their hyperrectangle has joint coverage
    >= 1 - alpha by Bonferroni (asymptotically). Returns the minimum
    NDIS eps over the rectangle.
    """
    if rng is None:
        rng = np.random.default_rng()
    in_scores = np.asarray(in_scores, dtype=float)
    out_scores = np.asarray(out_scores, dtype=float)
    samples = _bootstrap_param_samples(in_scores, out_scores, n_bootstrap, rng)

    a_marg = alpha / n_dim
    lo = np.quantile(samples, a_marg / 2.0, axis=0)
    hi = np.quantile(samples, 1.0 - a_marg / 2.0, axis=0)
    eps_lb, theta_min = _minimize_eps_over_box(lo, hi, delta, pool_variance=pool_variance)
    return {
        'eps_lb': eps_lb,
        'theta_min': theta_min,
        'box_lo': lo, 'box_hi': hi,
        'method': 'bootstrap_bonferroni',
        'samples': samples,
    }


# ---- (C) Simultaneous bootstrap (max-norm) box -- sharper than Bonferroni ----

def ndis_eps_lb_bootstrap_simultaneous_box(
    in_scores, out_scores, delta: float, *,
    alpha: float = 0.05, n_bootstrap: int = 2000,
    pool_variance: bool = False,
    rng: Optional[np.random.Generator] = None,
):
    """Lower bound via a simultaneous (max-norm) bootstrap rectangle.

    Bootstrap the parameter vector and form the hyperrectangle
        { theta : max_j |theta_j - hat theta_j| / s_j  <=  q }
    where s_j is the bootstrap std of coordinate j and q is the
    (1-alpha) quantile of M_b := max_j |theta_b,j - hat theta_j| / s_j
    over bootstrap replicates. By construction this rectangle has
    joint bootstrap coverage exactly 1-alpha, so it is uniformly
    tighter than the Bonferroni rectangle for the same alpha (no
    union-bound slack).
    """
    if rng is None:
        rng = np.random.default_rng()
    in_scores = np.asarray(in_scores, dtype=float)
    out_scores = np.asarray(out_scores, dtype=float)
    n_in, n_out = len(in_scores), len(out_scores)
    theta_hat = np.array([
        np.mean(in_scores), np.std(in_scores, ddof=1),
        np.mean(out_scores), np.std(out_scores, ddof=1),
    ])

    samples = _bootstrap_param_samples(in_scores, out_scores, n_bootstrap, rng)
    s = samples.std(axis=0, ddof=1)
    s = np.where(s > 0, s, 1.0)
    M = np.max(np.abs(samples - theta_hat[None, :]) / s[None, :], axis=1)
    q = float(np.quantile(M, 1.0 - alpha))
    lo = theta_hat - q * s
    hi = theta_hat + q * s
    eps_lb, theta_min = _minimize_eps_over_box(lo, hi, delta, pool_variance=pool_variance)
    return {
        'eps_lb': eps_lb,
        'theta_min': theta_min,
        'box_lo': lo, 'box_hi': hi,
        'q': q, 'theta_hat': theta_hat,
        'method': 'bootstrap_simultaneous_box',
        'samples': samples,
    }


# ---- (D) Joint-bootstrap ellipsoid (Mahalanobis) CR -------------------------

def ndis_eps_lb_bootstrap_ellipsoid(
    in_scores, out_scores, delta: float, *,
    alpha: float = 0.05, n_bootstrap: int = 2000,
    pool_variance: bool = False,
    rng: Optional[np.random.Generator] = None,
    n_starts: int = 12,
):
    """Lower bound via a joint elliptical CR estimated by bootstrap.

    Estimates the 4x4 bootstrap covariance Sigma_hat of theta and
    forms the ellipsoid
       C = { theta : (theta - hat theta)^T Sigma_hat^{-1} (theta - hat theta) <= r^2 }
    with r^2 = chi^2_{4, 1-alpha}. Asymptotically (theta is approximately
    Gaussian under the bootstrap) this is a (1-alpha)-coverage joint CR
    with no Bonferroni slack. We minimise eps(theta; delta) over C
    using sequential-quadratic programming.
    """
    if rng is None:
        rng = np.random.default_rng()
    in_scores = np.asarray(in_scores, dtype=float)
    out_scores = np.asarray(out_scores, dtype=float)
    theta_hat = np.array([
        np.mean(in_scores), np.std(in_scores, ddof=1),
        np.mean(out_scores), np.std(out_scores, ddof=1),
    ])
    samples = _bootstrap_param_samples(in_scores, out_scores, n_bootstrap, rng)
    Sigma = np.cov(samples, rowvar=False)
    # Tikhonov for safety; bootstrap covariance can be near-singular if
    # any coord has near-zero variance.
    Sigma = Sigma + 1e-12 * np.eye(4) * np.trace(Sigma) / 4.0
    Sigma_inv = np.linalg.inv(Sigma)
    r2 = chi2.ppf(1.0 - alpha, df=4)

    def constraint_fn(theta):
        d = theta - theta_hat
        return r2 - float(d @ Sigma_inv @ d)  # >= 0 inside ellipsoid

    cons = ({'type': 'ineq', 'fun': constraint_fn},)

    def obj(theta):
        return _eps_of_theta(theta, delta, pool_variance=pool_variance)

    # Bounds keep sigma's strictly positive.
    bounds = [(-np.inf, np.inf), (1e-9, np.inf), (-np.inf, np.inf), (1e-9, np.inf)]

    best_eps = float(obj(theta_hat))
    best_theta = theta_hat.copy()

    # Multi-start: include theta_hat plus draws from inside the ellipsoid.
    L = np.linalg.cholesky(Sigma)
    starts = [theta_hat]
    for _ in range(max(0, n_starts - 1)):
        z = rng.standard_normal(4)
        z = z / np.linalg.norm(z) * rng.uniform(0.0, math.sqrt(r2))
        starts.append(theta_hat + L @ z)

    for x0 in starts:
        try:
            res = minimize(obj, x0=x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 200, 'ftol': 1e-9})
            if res.success and res.fun < best_eps:
                best_eps = float(res.fun)
                best_theta = res.x
        except Exception:
            pass

    return {
        'eps_lb': max(0.0, best_eps),
        'theta_min': best_theta,
        'theta_hat': theta_hat,
        'Sigma': Sigma, 'r2': r2,
        'method': 'bootstrap_ellipsoid',
        'samples': samples,
    }


# ---- (E) Single dispatcher --------------------------------------------------

def ndis_eps_lb(
    in_scores, out_scores, delta: float, *,
    method: str = 'parametric_bonferroni',
    alpha: float = 0.05, n_bootstrap: int = 2000,
    pool_variance: bool = False, rng: Optional[np.random.Generator] = None,
):
    """Dispatch to one of the eps lower-bound constructions.

    method in {
        'parametric_bonferroni',
        'bootstrap_bonferroni',
        'bootstrap_simultaneous_box',
        'bootstrap_ellipsoid',
        'bootstrap_eps_quantile',     # current method, eps-quantile of bootstrap
    }
    """
    if method == 'parametric_bonferroni':
        return ndis_eps_lb_parametric_bonferroni(
            in_scores, out_scores, delta, alpha=alpha, pool_variance=pool_variance,
        )
    if method == 'bootstrap_bonferroni':
        return ndis_eps_lb_bootstrap_bonferroni(
            in_scores, out_scores, delta, alpha=alpha,
            n_bootstrap=n_bootstrap, pool_variance=pool_variance, rng=rng,
        )
    if method == 'bootstrap_simultaneous_box':
        return ndis_eps_lb_bootstrap_simultaneous_box(
            in_scores, out_scores, delta, alpha=alpha,
            n_bootstrap=n_bootstrap, pool_variance=pool_variance, rng=rng,
        )
    if method == 'bootstrap_ellipsoid':
        return ndis_eps_lb_bootstrap_ellipsoid(
            in_scores, out_scores, delta, alpha=alpha,
            n_bootstrap=n_bootstrap, pool_variance=pool_variance, rng=rng,
        )
    if method == 'bootstrap_eps_quantile':
        eps_lb = ndis_eps_lower_bound_with_ci(
            in_scores, out_scores, delta=delta, alpha=alpha,
            n_bootstrap=n_bootstrap, pool_variance=pool_variance, rng=rng,
        )
        return {'eps_lb': float(eps_lb), 'method': 'bootstrap_eps_quantile'}
    raise ValueError(f"unknown method: {method}")
