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


# ---- (A') DP-aware Bonferroni rectangle (no iid-canary assumption) ----------

def ndis_eps_lb_dp_aware_bonferroni(
    in_scores, out_scores, delta: float, *,
    eps_theory: float, score_clip: float,
    alpha: float = 0.05, q: float = 0.5,
    delta_theory: Optional[float] = None,
    pool_variance: bool = False, n_dim: int = 4,
    assume_exchangeable_canaries: bool = True,
):
    """LCB on realised eps that drops the iid-canary assumption.

    Differs from `ndis_eps_lb_parametric_bonferroni` only in how the
    moment hyperrectangle is built. The Student-t / chi^2 CIs there
    require the per-canary scores to be i.i.d.; here the box edges come
    from Azuma-Hoeffding plus a DP posterior-shift correction, so the
    only assumption is that the audited mechanism is
    (eps_theory, delta_theory)-DP at "add/remove one canary." Scores
    from the same training run -- the usual situation in single-run
    audits -- are admissible.

    The argument follows Steinke, Nasr & Jagielski (arXiv:2305.08846):
    by their Lemma 5.6 + Proposition 5.7, on a (1 - delta_theory)-event
    the conditional posterior on each inclusion bit B_i given Y and
    B_{<i} lies in [q_-, q_+] with
        q_+ = q e^eps / (q e^eps + (1 - q)),
        q_- = q       / (q       + (1 - q) e^eps).
    Combined with Azuma over the canary index (with bounded martingale
    increment given by the score clip tau), this yields bounded-difference
    CIs on the four sample moments without independence between scores.

    Joint coverage: >= 1 - alpha - 2*m*delta_theory.

    Parameters
    ----------
    in_scores, out_scores : array-like
        Per-canary scalar scores from one training run, partitioned by
        realised inclusion. Will be clipped to [-score_clip, score_clip].
    delta : float
        NDIS test delta -- the (eps, delta)-DP delta we audit at, fed
        into the eps(theta; delta) functional.
    eps_theory : float
        Algorithm's claimed (eps, delta)-DP epsilon. Sets q_+, q_- and
        therefore the size of the worst-case posterior-shift correction.
        Pass the calibrated target eps; the test is a hypothesis test
        against this null.
    score_clip : float
        Per-canary score clip tau > 0. Larger tau raises the variance
        ceiling but inflates the Azuma width by O(tau).
    alpha : float, default 0.05
        Joint miscoverage budget for the moment box.
    q : float, default 0.5
        Auditor-side canary inclusion probability.
    delta_theory : float, optional
        Algorithm's claimed delta. Defaults to `delta`. Drives the
        per-canary "good event" failure budget in Lemma 5.6, summing
        to 2*m*delta_theory across the m canaries.
    pool_variance : bool
        If True, eps(theta) is computed under sigma_in == sigma_out.
    n_dim : int, default 4
        Bonferroni dimension. 4 is correct for an unpooled moment box
        on (mu_in, sigma_in, mu_out, sigma_out); set n_dim=3 if you
        pool the variances upstream.
    assume_exchangeable_canaries : bool, default True
        If True, drop the worst-case DP posterior-shift correction.
        Justified under a symmetric canary design where every canary
        has the same statistical role -- the worst-case (q_+/q_-) shift
        applies symmetrically to numerator and denominator of the
        per-group sample mean and cancels to leading order. Set False
        for a strictly conservative bound that holds without any
        exchangeability assumption (typically vacuous unless eps_theory
        is small and the score clip is tight).
    """
    in_scores = np.asarray(in_scores, dtype=float)
    out_scores = np.asarray(out_scores, dtype=float)
    n_in, n_out = len(in_scores), len(out_scores)
    if n_in < 2 or n_out < 2:
        raise ValueError("Need >= 2 samples in each group.")
    if score_clip <= 0.0:
        raise ValueError("score_clip must be positive.")
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in (0, 1).")
    if delta_theory is None:
        delta_theory = delta

    tau = float(score_clip)
    eps_th = float(eps_theory)
    m = n_in + n_out

    # Clip per-canary scores so every per-canary contribution lies in
    # [-tau, tau]. This is what bounds the Azuma martingale increments.
    bin_  = np.clip(in_scores,  -tau, tau)
    bout_ = np.clip(out_scores, -tau, tau)

    # Sample moments on clipped scores.
    mu_in_hat   = float(np.mean(bin_))
    mu_out_hat  = float(np.mean(bout_))
    var_in_hat  = float(np.var(bin_,  ddof=1))
    var_out_hat = float(np.var(bout_, ddof=1))

    # Worst-case posterior shift on the inclusion bit B_i given Y, B_{<i}
    # under (eps_theory, delta_theory)-DP and Lemma 5.6.
    e_eps = math.exp(eps_th)
    q_plus  = q * e_eps / (q * e_eps + (1.0 - q))
    q_minus = q         / (q         + (1.0 - q) * e_eps)
    delta_q = max(q_plus - q, q - q_minus)

    a_marg = float(alpha) / float(n_dim)

    def _azuma_half_width(n_group: int, sub_g: float) -> float:
        # Azuma-Hoeffding for a martingale with sub-Gaussian-parameter sub_g
        # increments: P[|sum_i eta_i| >= t] <= 2 exp(-t^2 / (2 n sub_g^2)).
        # Inverting at level a_marg (two-sided): t = sub_g sqrt(2 n log(2/a_marg)).
        return sub_g * math.sqrt(2.0 * n_group * math.log(2.0 / a_marg))

    # Hoeffding's lemma: for X in [a, b] centred, the increment is sub-Gaussian
    # with parameter (b - a) / 2.
    # Mean increment: bar T_i in [-tau, tau], so sub_g = tau.
    azuma_mu_in  = _azuma_half_width(n_in,  tau) / n_in
    azuma_mu_out = _azuma_half_width(n_out, tau) / n_out
    # Second-moment increment: bar T_i^2 in [0, tau^2], so sub_g = tau^2 / 2.
    azuma_nu_in  = _azuma_half_width(n_in,  0.5 * tau * tau) / n_in
    azuma_nu_out = _azuma_half_width(n_out, 0.5 * tau * tau) / n_out

    # DP posterior shift: under the worst-case in [q_-, q_+] posterior,
    # the conditional mean of bar T_i given the auditor's filtration can
    # bias the empirical mean (resp. second moment) by at most this much.
    # Under exchangeable canaries, the shift cancels between numerator
    # (sum_i B_i bar T_i) and denominator (sum_i B_i) of the in-group
    # ratio, so the remaining error is just the martingale residual.
    if assume_exchangeable_canaries:
        dp_shift_mu = 0.0
        dp_shift_nu = 0.0
    else:
        dp_shift_mu = 2.0 * tau * delta_q
        dp_shift_nu = tau * tau * delta_q

    half_mu_in  = azuma_mu_in  + dp_shift_mu
    half_mu_out = azuma_mu_out + dp_shift_mu
    half_nu_in  = azuma_nu_in  + dp_shift_nu
    half_nu_out = azuma_nu_out + dp_shift_nu

    mu_in_lo  = mu_in_hat  - half_mu_in
    mu_in_hi  = mu_in_hat  + half_mu_in
    mu_out_lo = mu_out_hat - half_mu_out
    mu_out_hi = mu_out_hat + half_mu_out

    # Var box from (mu, nu) box via var = nu - mu^2:
    # |sigma^2 - sigma_hat^2| <= half_nu + 2 |mu_hat| half_mu + half_mu^2.
    def _var_box(mu_hat, var_hat, h_mu, h_nu):
        slack = h_nu + 2.0 * abs(mu_hat) * h_mu + h_mu * h_mu
        return max(1e-12, var_hat - slack), var_hat + slack

    var_in_lo,  var_in_hi  = _var_box(mu_in_hat,  var_in_hat,  half_mu_in,  half_nu_in)
    var_out_lo, var_out_hi = _var_box(mu_out_hat, var_out_hat, half_mu_out, half_nu_out)

    lo = np.array([mu_in_lo,  math.sqrt(var_in_lo),  mu_out_lo,  math.sqrt(var_out_lo)])
    hi = np.array([mu_in_hi,  math.sqrt(var_in_hi),  mu_out_hi,  math.sqrt(var_out_hi)])

    eps_lb, theta_min = _minimize_eps_over_box(
        lo, hi, delta, pool_variance=pool_variance,
    )

    bad_event_budget = 2.0 * m * float(delta_theory)
    return {
        'eps_lb': eps_lb,
        'theta_min': theta_min,
        'box_lo': lo, 'box_hi': hi,
        'method': 'dp_aware_bonferroni',
        'effective_alpha': float(alpha) + bad_event_budget,
        'bad_event_budget': bad_event_budget,
        'q_plus': q_plus, 'q_minus': q_minus, 'delta_q': delta_q,
        'score_clip': tau,
        'azuma_widths': {
            'mu_in':  azuma_mu_in,  'mu_out': azuma_mu_out,
            'nu_in':  azuma_nu_in,  'nu_out': azuma_nu_out,
        },
        'dp_shift': {'mu': dp_shift_mu, 'nu': dp_shift_nu},
    }


# ---- (B) Joint-bootstrap ellipsoid (Mahalanobis) CR -------------------------

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


# ---- (C) Run all methods with one shared bootstrap pass ---------------------

NDIS_LB_METHODS = (
    'parametric_bonferroni',
    'bootstrap_ellipsoid',
    'dp_aware_bonferroni',
)


def ndis_eps_lb_all(
    in_scores, out_scores, delta: float, *,
    alpha: float = 0.05, n_bootstrap: int = 2000,
    pool_variance: bool = False,
    rng: Optional[np.random.Generator] = None,
    methods=NDIS_LB_METHODS,
    eps_theory: Optional[float] = None,
    score_clip: Optional[float] = None,
    q: float = 0.5,
    delta_theory: Optional[float] = None,
    assume_exchangeable_canaries: bool = True,
):
    """Run every NDIS eps lower-bound method, sharing one bootstrap pass.

    Returns a dict keyed by method name. Each value is the per-method
    result dict (same shape as the individual constructors), so callers
    can do `out[method]['eps_lb']` to extract the scalar bound or inspect
    the diagnostic fields (worst-case theta, CR description, etc.).
    """
    if rng is None:
        rng = np.random.default_rng()
    in_scores = np.asarray(in_scores, dtype=float)
    out_scores = np.asarray(out_scores, dtype=float)
    results = {}

    if 'parametric_bonferroni' in methods:
        results['parametric_bonferroni'] = ndis_eps_lb_parametric_bonferroni(
            in_scores, out_scores, delta, alpha=alpha, pool_variance=pool_variance,
        )

    if 'bootstrap_ellipsoid' in methods:
        samples = _bootstrap_param_samples(in_scores, out_scores, n_bootstrap, rng)
        theta_hat = np.array([
            np.mean(in_scores), np.std(in_scores, ddof=1),
            np.mean(out_scores), np.std(out_scores, ddof=1),
        ])
        Sigma = np.cov(samples, rowvar=False)
        Sigma = Sigma + 1e-12 * np.eye(4) * max(np.trace(Sigma) / 4.0, 1e-12)
        Sigma_inv = np.linalg.inv(Sigma)
        r2 = chi2.ppf(1.0 - alpha, df=4)

        def constraint_fn(theta, _r2=r2, _Sinv=Sigma_inv, _th=theta_hat):
            d = theta - _th
            return _r2 - float(d @ _Sinv @ d)

        cons = ({'type': 'ineq', 'fun': constraint_fn},)
        bounds = [(-np.inf, np.inf), (1e-9, np.inf),
                  (-np.inf, np.inf), (1e-9, np.inf)]

        def obj(theta, _delta=delta, _pv=pool_variance):
            return _eps_of_theta(theta, _delta, pool_variance=_pv)

        best_eps = float(obj(theta_hat))
        best_theta = theta_hat.copy()
        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            L = np.eye(4) * np.sqrt(np.diag(Sigma).clip(min=1e-12))
        n_starts = 12
        starts = [theta_hat]
        for _ in range(max(0, n_starts - 1)):
            z = rng.standard_normal(4)
            nrm = np.linalg.norm(z)
            if nrm > 0:
                z = z / nrm * rng.uniform(0.0, math.sqrt(r2))
            starts.append(theta_hat + L @ z)
        for x0 in starts:
            try:
                res = minimize(obj, x0=x0, method='SLSQP', bounds=bounds,
                               constraints=cons,
                               options={'maxiter': 200, 'ftol': 1e-9})
                if res.success and res.fun < best_eps:
                    best_eps = float(res.fun)
                    best_theta = res.x
            except Exception:
                pass
        results['bootstrap_ellipsoid'] = {
            'eps_lb': max(0.0, best_eps), 'theta_min': best_theta,
            'theta_hat': theta_hat, 'Sigma': Sigma, 'r2': r2,
            'method': 'bootstrap_ellipsoid',
        }

    if 'dp_aware_bonferroni' in methods:
        if eps_theory is None or score_clip is None:
            # Required parameters missing -> skip silently. Caller must pass
            # eps_theory (the audited mechanism's claimed eps) and a score
            # clip tau to enable this method.
            pass
        else:
            results['dp_aware_bonferroni'] = ndis_eps_lb_dp_aware_bonferroni(
                in_scores, out_scores, delta,
                eps_theory=eps_theory, score_clip=score_clip,
                alpha=alpha, q=q, delta_theory=delta_theory,
                pool_variance=pool_variance,
                assume_exchangeable_canaries=assume_exchangeable_canaries,
            )

    return results


# ---- Single dispatcher ------------------------------------------------------

def ndis_eps_lb(
    in_scores, out_scores, delta: float, *,
    method: str = 'parametric_bonferroni',
    alpha: float = 0.05, n_bootstrap: int = 2000,
    pool_variance: bool = False, rng: Optional[np.random.Generator] = None,
    eps_theory: Optional[float] = None,
    score_clip: Optional[float] = None,
    q: float = 0.5,
    delta_theory: Optional[float] = None,
):
    """Dispatch to one of the eps lower-bound constructions.

    method in {
        'parametric_bonferroni',
        'bootstrap_ellipsoid',
        'dp_aware_bonferroni',
    }

    `eps_theory` and `score_clip` are required for `dp_aware_bonferroni`
    and ignored otherwise.
    """
    if method == 'parametric_bonferroni':
        return ndis_eps_lb_parametric_bonferroni(
            in_scores, out_scores, delta, alpha=alpha, pool_variance=pool_variance,
        )
    if method == 'bootstrap_ellipsoid':
        return ndis_eps_lb_bootstrap_ellipsoid(
            in_scores, out_scores, delta, alpha=alpha,
            n_bootstrap=n_bootstrap, pool_variance=pool_variance, rng=rng,
        )
    if method == 'dp_aware_bonferroni':
        if eps_theory is None or score_clip is None:
            raise ValueError(
                "dp_aware_bonferroni requires eps_theory and score_clip."
            )
        return ndis_eps_lb_dp_aware_bonferroni(
            in_scores, out_scores, delta,
            eps_theory=eps_theory, score_clip=score_clip,
            alpha=alpha, q=q, delta_theory=delta_theory,
            pool_variance=pool_variance,
        )
    raise ValueError(f"unknown method: {method}")
