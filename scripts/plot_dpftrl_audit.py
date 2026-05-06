#!/usr/bin/env python3
"""DP-FTRL audit visualization: bootstrap-CI box plot vs theoretical eps.

Reads N exp_dirs (one per target eps), bootstraps the IN/OUT canary
scores B times to produce a distribution of empirical-eps point estimates,
and plots the distribution alongside the theoretical-eps target. The point
of the figure is to demonstrate that our audit recovers the calibrated
privacy parameter, with the bootstrap spread quantifying uncertainty.

Usage:
    python scripts/plot_dpftrl_audit.py \\
        --exp-dirs ./data/dpftrl-scatter-eps1 \\
                   ./data/dpftrl-scatter-eps2 \\
                   ./data/dpftrl-scatter-eps4 \\
                   ./data/dpftrl-scatter-eps8

If --exp-dirs is given as a glob it is expanded by the shell as usual.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
src_dir = os.path.join(project_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from whitebox_auditing.ndis_1d import (
    estimate_mean_variance,
    _ndis_eps_from_moments,
    ndis_eps_lb_all,
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
    """Returns (in_scores, out_scores, target_eps, epoch). Prefers `optimal` files."""
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
# Audit (point estimate + LCB + bootstrap distribution)
# ---------------------------------------------------------------------------

def bootstrap_eps_distribution(in_sc, out_sc, *, delta, n_boot, rng,
                                pool_variance=True):
    """Resample with replacement, return n_boot empirical-eps point estimates.

    Each sample is the NDIS Gaussian-DP eps inverted from the bootstrap
    moments via Balle-Wang. The 2.5/97.5 percentiles of the returned array
    form a percentile bootstrap CI on the empirical eps estimate.
    """
    n_in, n_out = len(in_sc), len(out_sc)
    eps_samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        ib = in_sc[rng.integers(0, n_in,  n_in)]
        ob = out_sc[rng.integers(0, n_out, n_out)]
        mv = estimate_mean_variance(ib, ob)
        eps_samples[b] = _ndis_eps_from_moments(
            mv['in_mean'], mv['in_std'], mv['out_mean'], mv['out_std'],
            delta=delta, pool_variance=pool_variance,
        )
    return eps_samples


def point_estimate_and_lcb(in_sc, out_sc, *, delta, alpha, pool_variance=True):
    """Point estimate (no shrink) and parametric-Bonferroni LCB."""
    mv = estimate_mean_variance(in_sc, out_sc)
    eps_pt = _ndis_eps_from_moments(
        mv['in_mean'], mv['in_std'], mv['out_mean'], mv['out_std'],
        delta=delta, pool_variance=pool_variance,
    )
    res = ndis_eps_lb_all(
        in_sc, out_sc, delta=delta,
        alpha=alpha, pool_variance=pool_variance,
        n_bootstrap=2000,
    )
    return eps_pt, float(res['parametric_bonferroni']['eps_lb'])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(exp_dirs, *, delta, n_boot, alpha, fig_path, seed):
    rng = np.random.default_rng(seed)
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

        eps_dist = bootstrap_eps_distribution(
            in_sc, out_sc, delta=delta, n_boot=n_boot, rng=rng,
        )
        eps_pt, eps_lcb = point_estimate_and_lcb(
            in_sc, out_sc, delta=delta, alpha=alpha,
        )
        rows.append({
            'target': target_eps,
            'dist':   eps_dist,
            'point':  eps_pt,
            'lcb':    eps_lcb,
            'n_in':   len(in_sc),
            'n_out':  len(out_sc),
            'epoch':  epoch,
        })
        print(
            f'eps={target_eps:>5.2f}  '
            f'point={eps_pt:6.3f}  '
            f'LCB{int((1-alpha)*100):d}={eps_lcb:6.3f}  '
            f'boot[p2.5,p50,p97.5]=[{np.percentile(eps_dist,2.5):.3f}, '
            f'{np.percentile(eps_dist,50):.3f}, '
            f'{np.percentile(eps_dist,97.5):.3f}]  '
            f'(n_in={len(in_sc)}, n_out={len(out_sc)}, T={epoch})'
        )

    if not rows:
        sys.exit('No data loaded.')

    rows.sort(key=lambda r: r['target'])
    targets = [r['target'] for r in rows]
    dists   = [r['dist']   for r in rows]
    lcbs    = [r['lcb']    for r in rows]

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 14,
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.frameon': False,
    })
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    positions = np.arange(len(targets), dtype=float)

    # Theoretical eps target: red horizontal mark per group.
    for x, y in zip(positions, targets):
        ax.plot([x - 0.42, x + 0.42], [y, y], color='#cc4422',
                linewidth=2.6, zorder=4, solid_capstyle='round')
    ax.plot([], [], color='#cc4422', linewidth=2.6,
            label=r'Theoretical $\varepsilon$ (target)')

    # Bootstrap distribution as a box plot.
    ax.boxplot(
        dists, positions=positions, widths=0.55, showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='#cfe6f0', edgecolor='#00497D', linewidth=1.4),
        medianprops=dict(color='#00497D', linewidth=2.4),
        whiskerprops=dict(color='#00497D', linewidth=1.3),
        capprops=dict(color='#00497D', linewidth=1.3),
    )
    ax.plot([], [], color='#00497D', linewidth=2.4,
            label=r'Empirical $\varepsilon$ (bootstrap point-estimate dist.)')

    # Parametric-Bonferroni LCB markers (downward triangles).
    ax.scatter(positions, lcbs, marker='v', color='#00497D',
               s=80, zorder=6, edgecolor='white', linewidth=0.7,
               label=fr'{int((1-alpha)*100)}% LCB (parametric Bonferroni)')

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{t:g}' for t in targets])
    ax.set_xlabel(r'Theoretical $\varepsilon$ (calibrated target)')
    ax.set_ylabel(r'Empirical $\varepsilon$')
    ax.set_title(r'DP-FTRL audit: empirical $\varepsilon$ recovers theoretical target')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    fig.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'\nFigure saved to: {fig_path} (and .pdf)')


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='DP-FTRL audit: bootstrap-CI box plot vs theoretical eps.'
    )
    parser.add_argument('--exp-dirs', nargs='+', required=True,
                        help='One exp_dir per target epsilon.')
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='LCB miscoverage; 0.05 -> 95%% LCB.')
    parser.add_argument('--seed', type=int, default=0xBEE71E)
    parser.add_argument('--fig-path', type=str,
                        default=os.path.join(project_dir, 'fig', 'dp-ftrl-audit-bootstrap.png'))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.fig_path), exist_ok=True)
    make_plot(
        args.exp_dirs,
        delta=args.delta,
        n_boot=args.n_boot,
        alpha=args.alpha,
        fig_path=args.fig_path,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()