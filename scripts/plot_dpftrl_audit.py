#!/usr/bin/env python3
"""DP-FTRL audit visualization: provable parametric-Bonferroni CI on epsilon.

For each exp_dir (one per target epsilon), build the Bonferroni
hyperrectangle on theta = (mu_in, sigma_in, mu_out, sigma_out) at level
1 - alpha (Student-t for means, chi^2 for variances), then evaluate the
NDIS eps functional at every corner + an L-BFGS-B refinement to recover

    eps_LB = min_{theta in R} eps(theta; delta)
    eps_UB = max_{theta in R} eps(theta; delta)

By construction, P[theta_true in R] >= 1 - alpha, so
P[eps_LB <= eps(theta_true) <= eps_UB] >= 1 - alpha. This is a *provable*
joint (1-alpha) CI on the realized empirical epsilon. We plot it next
to the theoretical (calibrated) eps target.

Usage:
    python scripts/plot_dpftrl_audit.py \\
        --exp-dirs ./data/dpftrl-scatter-eps1 \\
                   ./data/dpftrl-scatter-eps2 \\
                   ./data/dpftrl-scatter-eps4 \\
                   ./data/dpftrl-scatter-eps8
"""

import os
import sys
import json
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
src_dir = os.path.join(project_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from scipy.optimize import minimize
from scipy.stats import chi2, t as student_t

from whitebox_auditing.ndis_1d import (
    estimate_mean_variance,
    _ndis_eps_from_moments,
    _eps_of_theta,
    ndis_eps_lb_parametric_bonferroni,
)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _latest_epoch(exp_dir, prefix):
    epochs = sorted(set(
        int(f.split('_')[-1].replace('.csv', ''))
        for f in os.listdir(exp_dir)
        if f.startswith(prefix) and f.endswith('.csv')
    ))
    return epochs[-1] if epochs else None


def load_exp(exp_dir):
    """Returns (in_scores, out_scores, target_eps, epoch). Prefers `optimal` files.

    NDIS eps is invariant to a common affine rescaling, so optimal vs ndis
    files give the same answer. We still prefer `optimal` for interpretability.
    """
    if not os.path.isdir(exp_dir):
        return None
    for in_prefix in ('in_scores_optimal_', 'in_scores_ndis_'):
        epoch = _latest_epoch(exp_dir, in_prefix)
        if epoch is None:
            continue
        out_prefix = 'out_' + in_prefix[len('in_'):]
        in_csv  = os.path.join(exp_dir, f'{in_prefix}{epoch:06d}.csv')
        out_csv = os.path.join(exp_dir, f'{out_prefix}{epoch:06d}.csv')
        if os.path.isfile(in_csv) and os.path.isfile(out_csv):
            in_sc  = np.loadtxt(in_csv,  delimiter=',')
            out_sc = np.loadtxt(out_csv, delimiter=',')
            target = None
            hp = os.path.join(exp_dir, 'hparams.json')
            if os.path.isfile(hp):
                with open(hp) as f:
                    target = float(json.load(f).get('epsilon'))
            return in_sc, out_sc, target, epoch
    return None


# ---------------------------------------------------------------------------
# Provable LCB + matching UCB on the same Bonferroni rectangle.
# ---------------------------------------------------------------------------

def _bonferroni_rectangle(in_scores, out_scores, *, alpha, n_dim=4):
    """Per-coord (1 - alpha/n_dim) CIs on (mu_in, sigma_in, mu_out, sigma_out).

    Mirrors the rectangle used inside `ndis_eps_lb_parametric_bonferroni`.
    Bonferroni gives joint coverage >= 1 - alpha.
    """
    in_scores = np.asarray(in_scores, dtype=float)
    out_scores = np.asarray(out_scores, dtype=float)
    n_in, n_out = len(in_scores), len(out_scores)

    a_marg = alpha / n_dim
    mu_in_hat  = float(np.mean(in_scores))
    mu_out_hat = float(np.mean(out_scores))
    s2_in  = float(np.var(in_scores, ddof=1))
    s2_out = float(np.var(out_scores, ddof=1))

    t_in  = student_t.ppf(1.0 - a_marg / 2.0, df=n_in - 1)
    t_out = student_t.ppf(1.0 - a_marg / 2.0, df=n_out - 1)
    mu_in_lo  = mu_in_hat  - t_in  * math.sqrt(s2_in  / n_in)
    mu_in_hi  = mu_in_hat  + t_in  * math.sqrt(s2_in  / n_in)
    mu_out_lo = mu_out_hat - t_out * math.sqrt(s2_out / n_out)
    mu_out_hi = mu_out_hat + t_out * math.sqrt(s2_out / n_out)

    chi2_in_lo  = chi2.ppf(a_marg / 2.0,       df=n_in - 1)
    chi2_in_hi  = chi2.ppf(1.0 - a_marg / 2.0, df=n_in - 1)
    chi2_out_lo = chi2.ppf(a_marg / 2.0,       df=n_out - 1)
    chi2_out_hi = chi2.ppf(1.0 - a_marg / 2.0, df=n_out - 1)
    sig_in_lo  = math.sqrt((n_in  - 1) * s2_in  / chi2_in_hi)
    sig_in_hi  = math.sqrt((n_in  - 1) * s2_in  / chi2_in_lo)
    sig_out_lo = math.sqrt((n_out - 1) * s2_out / chi2_out_hi)
    sig_out_hi = math.sqrt((n_out - 1) * s2_out / chi2_out_lo)

    lo = np.array([mu_in_lo, sig_in_lo, mu_out_lo, sig_out_lo])
    hi = np.array([mu_in_hi, sig_in_hi, mu_out_hi, sig_out_hi])
    lo = np.maximum(lo, [-(np.inf), 1e-12, -(np.inf), 1e-12])
    return lo, hi


def _maximize_eps_over_box(lo, hi, *, delta, pool_variance, n_starts=5,
                           rng=None):
    """Mirror of ndis_1d._minimize_eps_over_box but for the supremum.

    Corners + L-BFGS-B refinement. eps is monotone in each coord under the
    standard equal-variance setup, so the corner search is typically exact.
    """
    if rng is None:
        rng = np.random.default_rng(1)
    best_eps = -np.inf
    best_theta = None
    for mask in range(16):
        corner = np.where(np.array([(mask >> b) & 1 for b in range(4)], dtype=bool), hi, lo)
        e = _eps_of_theta(corner, delta, pool_variance=pool_variance)
        if e > best_eps:
            best_eps, best_theta = e, corner

    bounds = list(zip(lo.tolist(), hi.tolist()))
    def neg_eps(theta):
        return -_eps_of_theta(theta, delta, pool_variance=pool_variance)
    starts = [best_theta] + [rng.uniform(lo, hi) for _ in range(max(0, n_starts - 1))]
    for x0 in starts:
        try:
            res = minimize(neg_eps, x0=x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 50, 'ftol': 1e-9})
            if -res.fun > best_eps:
                best_eps = float(-res.fun)
                best_theta = res.x
        except Exception:
            pass
    return float(max(0.0, best_eps)), best_theta


def parametric_bonferroni_ci(in_sc, out_sc, *, delta, alpha, pool_variance=True):
    """Provable (1 - alpha) CI on realized eps via the Bonferroni rectangle.

    Returns dict with eps_lcb (provable lower bound), eps_ucb (provable
    upper bound) and the unshrunk point estimate.
    """
    lcb_res = ndis_eps_lb_parametric_bonferroni(
        in_sc, out_sc, delta=delta, alpha=alpha, pool_variance=pool_variance,
    )
    eps_lcb = float(lcb_res['eps_lb'])
    lo, hi = lcb_res['box_lo'], lcb_res['box_hi']
    eps_ucb, _ = _maximize_eps_over_box(
        lo, hi, delta=delta, pool_variance=pool_variance,
    )

    mv = estimate_mean_variance(in_sc, out_sc)
    eps_pt = _ndis_eps_from_moments(
        mv['in_mean'], mv['in_std'], mv['out_mean'], mv['out_std'],
        delta=delta, pool_variance=pool_variance,
    )
    return {
        'eps_lcb':   eps_lcb,
        'eps_ucb':   eps_ucb,
        'eps_point': eps_pt,
        'box_lo':    lo,
        'box_hi':    hi,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(exp_dirs, *, delta, alpha, pool_variance, fig_path):
    rows = []
    for d in exp_dirs:
        loaded = load_exp(d)
        if loaded is None:
            print(f'skip (no usable scores): {d}')
            continue
        in_sc, out_sc, target_eps, epoch = loaded
        if target_eps is None:
            print(f'skip (no hparams.json): {d}')
            continue

        ci = parametric_bonferroni_ci(
            in_sc, out_sc, delta=delta, alpha=alpha, pool_variance=pool_variance,
        )
        rows.append({
            'target': target_eps,
            'lcb':    ci['eps_lcb'],
            'ucb':    ci['eps_ucb'],
            'point':  ci['eps_point'],
            'n_in':   len(in_sc),
            'n_out':  len(out_sc),
            'epoch':  epoch,
        })
        cov = int((1 - alpha) * 100)
        print(
            f'eps={target_eps:>5.2f}  '
            f'point={ci["eps_point"]:6.3f}  '
            f'{cov}%-CI=[{ci["eps_lcb"]:6.3f}, {ci["eps_ucb"]:6.3f}]  '
            f'(n_in={len(in_sc)}, n_out={len(out_sc)}, T={epoch})'
        )

    if not rows:
        sys.exit('No data loaded.')

    rows.sort(key=lambda r: r['target'])
    targets = np.array([r['target'] for r in rows])
    lcbs    = np.array([r['lcb']    for r in rows])
    ucbs    = np.array([r['ucb']    for r in rows])
    points  = np.array([r['point']  for r in rows])

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 22,
        'axes.titlesize': 22,
        'axes.labelsize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.frameon': False,
        'legend.fontsize': 20,
    })
    fig, ax = plt.subplots(figsize=(11, 6.5))
    positions = np.arange(len(targets), dtype=float)

    # Theoretical (calibrated) eps target.
    for x, y in zip(positions, targets):
        ax.plot([x - 0.42, x + 0.42], [y, y], color='#cc4422',
                linewidth=2.6, zorder=4, solid_capstyle='round')
    ax.plot([], [], color='#cc4422', linewidth=2.6,
            label=r'Theoretical $\varepsilon$ (calibrated target)')

    # Provable Bonferroni CI as an error bar centered on the point estimate.
    cov = int((1 - alpha) * 100)
    err_lo = np.maximum(0.0, points - lcbs)
    err_hi = np.maximum(0.0, ucbs - points)
    ax.errorbar(
        positions, points,
        yerr=np.vstack([err_lo, err_hi]),
        fmt='o', color='#00497D', ecolor='#00497D',
        elinewidth=2.0, capsize=6, capthick=1.6, markersize=8,
        markerfacecolor='#cfe6f0', markeredgewidth=1.4,
        zorder=5,
        label=fr'NDIS $\hat\varepsilon$ with provable {cov}\% CI '
              r'(parametric Bonferroni)',
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{t:g}' for t in targets])
    ax.set_xlabel(r'Theoretical $\varepsilon$ (calibrated target)')
    ax.set_ylabel(r'Empirical $\varepsilon$')
    ax.set_title(r'DP-FTRL audit: provable confidence interval recovers theoretical $\varepsilon$')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    fig.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'\nFigure saved to: {fig_path} (and .pdf)')


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='DP-FTRL audit: provable Bonferroni CI on empirical eps vs theoretical eps.'
    )
    parser.add_argument('--exp-dirs', nargs='+', required=True,
                        help='One exp_dir per target epsilon.')
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Joint miscoverage; 0.05 -> 95%% CI.')
    parser.add_argument('--pool-variance', action='store_true', default=True)
    parser.add_argument('--no-pool-variance', dest='pool_variance', action='store_false')
    parser.add_argument('--fig-path', type=str,
                        default=os.path.join(project_dir, 'fig', 'dp-ftrl-audit-bonferroni-ci.png'))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.fig_path), exist_ok=True)
    make_plot(
        args.exp_dirs,
        delta=args.delta,
        alpha=args.alpha,
        pool_variance=args.pool_variance,
        fig_path=args.fig_path,
    )


if __name__ == '__main__':
    main()
