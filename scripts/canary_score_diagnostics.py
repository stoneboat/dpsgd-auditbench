"""
Empirical diagnostics for the i.i.d. Gaussian per-canary score assumption
used in the one-run DP-SGD auditor.

Two assumptions are tested:
  (A) Cross-canary independence: scores from different canaries (within the
      same neighboring world b) are approximately uncorrelated.
  (B) Marginal Gaussianity: each canary score s_i^{(b)} is approximately
      Gaussian, justifying the parametric confidence region.

INPUTS
------
scores_absent  : np.ndarray of shape (n_runs, m_absent)
    Each row is one DP-SGD run; each column is one absent-canary score s_i^{(0)}.
scores_present : np.ndarray of shape (n_runs, m_present)
    Same, for present-canary scores s_i^{(1)}.

If you only have a single run (the common one-run setting), the
cross-canary correlation diagnostic is computed by treating the m canaries
within that run as the sample. The Gaussianity diagnostic uses the empirical
distribution of scores within the run.

Recommended diagnostics to report in the paper:
  1. Pairwise correlation summary (mean, max, 99th percentile of |corr|).
  2. Shapiro-Wilk and Anderson-Darling p-values for marginal Gaussianity.
  3. Q-Q plot vs theoretical Gaussian.
  4. Score-variance match: empirical Var(s^{(0)}) vs theoretical sigma^2
     under Model 1.
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# (A) Cross-canary independence
# ---------------------------------------------------------------------------

def cross_canary_correlation(scores, max_pairs=10000, rng=None):
    """
    Estimate pairwise correlations between canary scores.

    For a single-run setting (scores: shape (m,)), we cannot estimate
    pairwise correlations directly — there is only one realization per
    canary. Instead, run the auditor B times (small, e.g. B=20-50) on
    independent training data and treat the B realizations of each canary
    score as the sample. Then `scores` should have shape (B, m), and we
    estimate Corr(s_i, s_j) across the B replicates.

    Parameters
    ----------
    scores : (B, m) array
        B independent runs of the auditor, m canaries each.
    max_pairs : int
        Subsample at most this many (i, j) pairs to keep the diagnostic
        cheap when m is large. m * (m-1) / 2 can be huge.
    rng : np.random.Generator or None

    Returns
    -------
    dict with keys: 'n_pairs', 'mean_abs_corr', 'max_abs_corr',
                    'p99_abs_corr', 'corr_samples' (the absolute correlations)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    B, m = scores.shape
    if B < 10:
        raise ValueError(
            f"Need at least ~10 independent runs to estimate correlations; got B={B}. "
            "If you only have one run, use within_run_orthogonality_check() instead."
        )

    # Subsample pairs if m is large
    total_pairs = m * (m - 1) // 2
    if total_pairs > max_pairs:
        pair_idx = rng.choice(total_pairs, size=max_pairs, replace=False)
        # Convert linear index to (i, j) with i < j
        # (we use a simple inverse; fine for the sizes we care about)
        i_arr = np.zeros(max_pairs, dtype=int)
        j_arr = np.zeros(max_pairs, dtype=int)
        for k, p in enumerate(pair_idx):
            # find i, j such that p = i*m - i*(i+1)/2 + (j - i - 1) (no need for speed)
            # easier: just sample i, j directly with i != j and i < j
            pass
        # simpler: just sample (i, j) directly
        i_arr = rng.integers(0, m, size=max_pairs)
        j_arr = rng.integers(0, m, size=max_pairs)
        keep = i_arr != j_arr
        i_arr, j_arr = i_arr[keep], j_arr[keep]
        # enforce i < j
        lo = np.minimum(i_arr, j_arr)
        hi = np.maximum(i_arr, j_arr)
        i_arr, j_arr = lo, hi
    else:
        i_arr, j_arr = np.triu_indices(m, k=1)

    # Standardize each column (canary) once
    centered = scores - scores.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, keepdims=True, ddof=1)
    std[std == 0] = 1.0
    normalized = centered / std

    # Pairwise correlation = mean of product of normalized columns
    products = normalized[:, i_arr] * normalized[:, j_arr]
    corrs = products.mean(axis=0)
    abs_corrs = np.abs(corrs)

    return {
        "n_pairs": len(corrs),
        "mean_abs_corr": float(abs_corrs.mean()),
        "max_abs_corr": float(abs_corrs.max()),
        "p99_abs_corr": float(np.quantile(abs_corrs, 0.99)),
        "corr_samples": corrs,
    }


def within_run_orthogonality_check(canary_directions):
    """
    Single-run sanity check: verify canary directions u^{(i)} are
    approximately orthogonal. This is a necessary (not sufficient)
    condition for the projected DP noise to be uncorrelated across
    canaries.

    Parameters
    ----------
    canary_directions : (m, d) array
        Unit vectors u^{(i)} used to project the noisy gradient at the
        relevant step (or averaged across steps).

    Returns
    -------
    dict with summary of |<u_i, u_j>| for i != j.

    Theory: for random unit vectors in d dimensions, |<u_i, u_j>| concentrates
    around 1/sqrt(d). For d ~ 1e6 (typical model size), expect ~1e-3.
    """
    m, d = canary_directions.shape
    # Normalize defensively
    norms = np.linalg.norm(canary_directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = canary_directions / norms
    # Gram matrix off-diagonal
    G = U @ U.T  # (m, m)
    iu = np.triu_indices(m, k=1)
    inner = G[iu]
    return {
        "m": m,
        "d": d,
        "expected_random_baseline": 1.0 / np.sqrt(d),
        "mean_abs_inner": float(np.abs(inner).mean()),
        "max_abs_inner": float(np.abs(inner).max()),
        "p99_abs_inner": float(np.quantile(np.abs(inner), 0.99)),
    }


# ---------------------------------------------------------------------------
# (B) Marginal Gaussianity
# ---------------------------------------------------------------------------

def gaussianity_tests(scores, label=""):
    """
    Test whether a 1-D sample of canary scores is consistent with Gaussian.

    Parameters
    ----------
    scores : (N,) array
        N canary scores from world b (e.g., all absent-canary scores from
        one or more runs, pooled).
    label : str
        For printing.

    Returns
    -------
    dict with Shapiro-Wilk and Anderson-Darling results.

    Notes
    -----
    - Shapiro-Wilk is sensitive but capped at N=5000 in scipy.
    - Anderson-Darling is recommended for larger N.
    - Kolmogorov-Smirnov against fitted N(mu_hat, sigma_hat^2) is NOT a
      proper test (the parameters are estimated), so we use Lilliefors
      via Anderson-Darling with `dist='norm'`.
    """
    scores = np.asarray(scores).ravel()
    N = len(scores)

    out = {"label": label, "N": N,
           "mean": float(scores.mean()),
           "std": float(scores.std(ddof=1)),
           "skew": float(stats.skew(scores)),
           "excess_kurtosis": float(stats.kurtosis(scores))}

    # Shapiro-Wilk (use up to 5000 samples)
    if N <= 5000:
        sw_stat, sw_p = stats.shapiro(scores)
    else:
        rng = np.random.default_rng(0)
        sub = rng.choice(scores, size=5000, replace=False)
        sw_stat, sw_p = stats.shapiro(sub)
    out["shapiro_W"] = float(sw_stat)
    out["shapiro_p"] = float(sw_p)

    # Anderson-Darling for normality (Lilliefors-style: parameters estimated)
    ad_result = stats.anderson(scores, dist="norm")
    out["anderson_stat"] = float(ad_result.statistic)
    # critical values are at significance levels [15, 10, 5, 2.5, 1] %
    out["anderson_critical_5pct"] = float(ad_result.critical_values[2])
    out["anderson_reject_5pct"] = bool(
        ad_result.statistic > ad_result.critical_values[2]
    )

    return out


def qq_plot_data(scores):
    """
    Return (theoretical_quantiles, sorted_scores) for a Q-Q plot vs
    standard Gaussian. Standardizes scores first.

    Use with matplotlib:
        tq, sq = qq_plot_data(scores)
        plt.plot(tq, sq, '.')
        plt.plot([tq.min(), tq.max()], [tq.min(), tq.max()], 'r-')
    """
    scores = np.asarray(scores).ravel()
    z = (scores - scores.mean()) / scores.std(ddof=1)
    sorted_z = np.sort(z)
    n = len(sorted_z)
    # Use (i - 0.5) / n plotting positions (Hazen)
    p = (np.arange(1, n + 1) - 0.5) / n
    theoretical_q = stats.norm.ppf(p)
    return theoretical_q, sorted_z


# ---------------------------------------------------------------------------
# Variance match against Model 1
# ---------------------------------------------------------------------------

def model1_variance_match(scores_absent, scores_present, sigma, q, C):
    """
    Compare empirical score variances to Model 1 predictions:
        Var[S^{(0)}] = sigma^2
        Var[S^{(1)}] = sigma^2 + q(1-q) C^2

    Parameters
    ----------
    scores_absent  : array of absent-canary scores
    scores_present : array of present-canary scores
    sigma, q, C    : DP-SGD parameters used to build the theoretical predictions

    Returns
    -------
    dict with empirical and theoretical variances, plus the ratios.
    """
    var0_emp = float(np.var(scores_absent, ddof=1))
    var1_emp = float(np.var(scores_present, ddof=1))
    var0_thy = sigma ** 2
    var1_thy = sigma ** 2 + q * (1 - q) * C ** 2

    return {
        "Var_absent_empirical": var0_emp,
        "Var_absent_theoretical": var0_thy,
        "ratio_absent": var0_emp / var0_thy,
        "Var_present_empirical": var1_emp,
        "Var_present_theoretical": var1_thy,
        "ratio_present": var1_emp / var1_thy,
        "mean_present_empirical": float(np.mean(scores_present)),
        "mean_present_theoretical_sqrtT_qC": "compute as sqrt(T) * q * C",
    }


# ---------------------------------------------------------------------------
# One-call summary
# ---------------------------------------------------------------------------

def run_all_diagnostics(scores_absent, scores_present,
                        canary_directions=None,
                        sigma=None, q=None, C=None,
                        verbose=True):
    """
    Run all diagnostics and pretty-print results.

    scores_absent, scores_present : 1-D or 2-D arrays
        If 1-D (single run): pooled canary scores.
        If 2-D (B, m): B runs by m canaries; cross-canary correlation
        across runs is computed.
    canary_directions : (m, d) array, optional
        For the orthogonality sanity check.
    sigma, q, C : floats, optional
        DP-SGD parameters for the variance-match check.
    """
    print("=" * 70)
    print("PER-CANARY SCORE DIAGNOSTICS")
    print("=" * 70)

    # Gaussianity (always available)
    flat_abs = np.asarray(scores_absent).ravel()
    flat_pre = np.asarray(scores_present).ravel()

    print("\n[1] Marginal Gaussianity tests")
    print("-" * 70)
    for label, sc in [("absent  (b=0)", flat_abs), ("present (b=1)", flat_pre)]:
        res = gaussianity_tests(sc, label=label)
        print(f"  {label}: N={res['N']}, mean={res['mean']:.4f}, "
              f"std={res['std']:.4f}, skew={res['skew']:.4f}, "
              f"excess kurt={res['excess_kurtosis']:.4f}")
        print(f"    Shapiro-Wilk:    W={res['shapiro_W']:.4f}, "
              f"p={res['shapiro_p']:.4g}")
        print(f"    Anderson-Darling: A^2={res['anderson_stat']:.4f}, "
              f"crit_5%={res['anderson_critical_5pct']:.4f}, "
              f"reject@5%={res['anderson_reject_5pct']}")

    # Cross-canary correlation (needs 2-D)
    print("\n[2] Cross-canary independence")
    print("-" * 70)
    sa = np.asarray(scores_absent)
    sp = np.asarray(scores_present)
    if sa.ndim == 2 and sa.shape[0] >= 10:
        for label, S in [("absent  (b=0)", sa), ("present (b=1)", sp)]:
            res = cross_canary_correlation(S)
            print(f"  {label}: pairs={res['n_pairs']}, "
                  f"mean|corr|={res['mean_abs_corr']:.4f}, "
                  f"max|corr|={res['max_abs_corr']:.4f}, "
                  f"99th pct |corr|={res['p99_abs_corr']:.4f}")
        # Reference: under true independence with B runs, |corr| ~ 1/sqrt(B-2)
        B = sa.shape[0]
        print(f"  (Reference: under true independence, |corr| ~ 1/sqrt(B-2) "
              f"= {1.0 / np.sqrt(max(B - 2, 1)):.4f})")
    else:
        print("  Skipped: needs >=10 independent runs (scores shape (B, m)).")

    if canary_directions is not None:
        res = within_run_orthogonality_check(canary_directions)
        print(f"  Canary direction orthogonality (m={res['m']}, d={res['d']}):")
        print(f"    mean |<u_i, u_j>| = {res['mean_abs_inner']:.4e}")
        print(f"    max  |<u_i, u_j>| = {res['max_abs_inner']:.4e}")
        print(f"    random baseline 1/sqrt(d) = "
              f"{res['expected_random_baseline']:.4e}")

    # Variance match
    if sigma is not None and q is not None and C is not None:
        print("\n[3] Variance match against Model 1")
        print("-" * 70)
        res = model1_variance_match(flat_abs, flat_pre, sigma, q, C)
        print(f"  Var[S^(0)]: empirical={res['Var_absent_empirical']:.4f}, "
              f"theoretical={res['Var_absent_theoretical']:.4f}, "
              f"ratio={res['ratio_absent']:.4f}")
        print(f"  Var[S^(1)]: empirical={res['Var_present_empirical']:.4f}, "
              f"theoretical={res['Var_present_theoretical']:.4f}, "
              f"ratio={res['ratio_present']:.4f}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic data matching Model 1 to verify the diagnostics work
    rng = np.random.default_rng(0)
    sigma, q, C, T = 1.0, 0.05, 1.0, 2500
    B, m = 30, 200  # 30 runs, 200 canaries each

    # Absent: N(0, sigma^2)
    sa = rng.normal(0, sigma, size=(B, m))
    # Present: sum_t (B_t C + Z_t) / sqrt(T), B_t Bernoulli(q)
    Bsamples = rng.binomial(1, q, size=(B, m, T))
    Zsamples = rng.normal(0, sigma, size=(B, m, T))
    sp = (Bsamples * C + Zsamples).sum(axis=2) / np.sqrt(T)

    run_all_diagnostics(sa, sp, sigma=sigma, q=q, C=C)
