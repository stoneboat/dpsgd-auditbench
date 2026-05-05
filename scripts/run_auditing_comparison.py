#!/usr/bin/env python3
"""
Run all 3 auditing methods on saved whitebox scores and plot comparison.

Methods:
  1. Steinke et al. 2023 (one-run, Theorem 5.2) — uses raw sum scores
  2. Mahloujifar et al. 2024 (f-DP) — uses raw sum scores
  3. NDIS (normal distribution indistinguishability spectrum) — uses normalized scores

Usage:
  # Single experiment directory:
  python scripts/run_auditing_comparison.py --exp-dir ./data/mislabeled-canaries-<seed>-5000-0.5-cifar10

  # Multiple experiment directories (one per epsilon), final-epoch bar chart:
  python scripts/run_auditing_comparison.py --exp-dirs ./data/exp_eps1 ./data/exp_eps2 ./data/exp_eps4 ./data/exp_eps8

  # Sample-complexity sweep on a single experiment:
  python scripts/run_auditing_comparison.py --complexity --exp-dir ./data/mislabeled-canaries-<seed>-5000-0.5-cifar10
"""

import sys
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

_RC = {
    'font.family': 'DejaVu Sans',
    'font.size': 14,
    'axes.titlesize': 12,
    'axes.labelsize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.color': '#cccccc',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.7,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.frameon': False,
    'legend.fontsize': 14,
    'figure.dpi': 120,
}

_STYLE = {
    'upper':   {'color': '#555555', 'marker': '',  'linestyle': (0, (3, 5, 1, 5)),  'linewidth': 1.4, 'markersize': 0,  'zorder': 1, 'label': 'Theoretical (upper bound)'},
    'steinke': {'color': '#ff7f0e', 'marker': 'o', 'linestyle': '--',  'linewidth': 2.4, 'markersize': 7,  'zorder': 2, 'label': 'Steinke et al. 2023'},
    'fdp':     {'color': '#2ca02c', 'marker': 's', 'linestyle': '--',  'linewidth': 2.4, 'markersize': 7,  'zorder': 3, 'label': 'Mahloujifar et al. 2024 (f-DP)'},
    'ndis':    {'color': '#1f77b4', 'marker': 'D', 'linestyle': '-',  'linewidth': 2.4, 'markersize': 7,  'zorder': 4, 'label': 'This paper'},
}


def _plot_method(ax, x, y, key):
    s = _STYLE[key]
    ax.plot(x, y,
            color=s['color'], marker=s['marker'], linestyle=s['linestyle'],
            linewidth=s['linewidth'], markersize=s['markersize'],
            label=s['label'], zorder=s['zorder'],
            markerfacecolor=s['color'], markeredgecolor='white', markeredgewidth=0.6)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)

from auditing import CanaryScoreAuditor
from whitebox_auditing.ndis_1d import (
    ndis_eps_from_delta_1d_brentq,
    ndis_eps_lower_bound_with_ci,
    estimate_mean_variance,
)


def _resolve_score_path(exp_dir, kind, side, epoch):
    """Resolve raw-score path: prefer `_sum_` (DP-SGD), fall back to `_optimal_`
    (DP-FTRL ancestor-sum). Both are scalar per-canary scores with the same
    in-vs-out semantics for the Steinke/Mahloujifar one-run estimators.
    """
    candidates = [
        os.path.join(exp_dir, f'{side}_scores_{kind}_{epoch:06d}.csv'),
        os.path.join(exp_dir, f'{side}_scores_optimal_{epoch:06d}.csv'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def audit_epoch(exp_dir, epoch, delta, significance):
    """Run all 3 auditing methods for a given epoch. Returns dict of results."""
    in_sum_path = _resolve_score_path(exp_dir, 'sum', 'in', epoch)
    out_sum_path = _resolve_score_path(exp_dir, 'sum', 'out', epoch)
    in_ndis_path = os.path.join(exp_dir, f'in_scores_ndis_{epoch:06d}.csv')
    out_ndis_path = os.path.join(exp_dir, f'out_scores_ndis_{epoch:06d}.csv')
    privacy_path = os.path.join(exp_dir, f'privacy_params_{epoch:06d}.csv')

    paths = [in_sum_path, out_sum_path, in_ndis_path, out_ndis_path]
    if not all(p is not None and os.path.isfile(p) for p in paths):
        return None

    in_sum = np.loadtxt(in_sum_path, delimiter=',')
    out_sum = np.loadtxt(out_sum_path, delimiter=',')
    in_ndis = np.loadtxt(in_ndis_path, delimiter=',')
    out_ndis = np.loadtxt(out_ndis_path, delimiter=',')

    # Theoretical upper bound
    if os.path.isfile(privacy_path):
        pp = np.loadtxt(privacy_path, delimiter=',', skiprows=1)
        eps_upper = float(pp[0]) if pp.ndim == 1 else float(pp[0, 0])
    else:
        eps_upper = float('nan')

    # Steinke et al. 2023 and Mahloujifar et al. 2024.
    auditor = CanaryScoreAuditor(in_sum, out_sum)
    eps_steinke, _ = auditor._epsilon_one_run_all_thresholds(
        significance=significance, delta=delta, one_sided=True, threshold=None,
        use_fdp=False,
    )
    eps_fdp, _ = auditor._epsilon_one_run_all_thresholds(
        significance=significance, delta=delta, one_sided=True, threshold=None,
        use_fdp=True,
    )

    # NDIS: use the bootstrap CI lower bound with pool_variance=True
    try:
        eps_ndis = ndis_eps_lower_bound_with_ci(
            in_ndis, out_ndis, delta=delta,
            alpha=significance, n_bootstrap=2000,
            pool_variance=False,
        )
    except (ValueError, RuntimeError) as e:
        print(f"  NDIS failed ({e}), setting to 0")
        eps_ndis = 0.0

    return {
        'upper': eps_upper,
        'steinke': eps_steinke,
        'fdp': eps_fdp,
        'ndis': eps_ndis,
    }


def get_final_epoch(exp_dir):
    """Find the largest epoch with score files. Accept either `_sum_`
    (DP-SGD) or `_optimal_` (DP-FTRL) raw-score filenames.
    """
    epochs = sorted(set(
        int(f.split('_')[-1].replace('.csv', ''))
        for f in os.listdir(exp_dir)
        if (f.startswith('in_scores_sum_') or f.startswith('in_scores_optimal_'))
        and f.endswith('.csv')
    ))
    return epochs[-1] if epochs else None


def get_target_epsilon(exp_dir):
    """Read target epsilon from hparams.json."""
    hparams_path = os.path.join(exp_dir, 'hparams.json')
    if os.path.isfile(hparams_path):
        with open(hparams_path) as f:
            return json.load(f).get('epsilon')
    return None


def run_single(exp_dir, delta, significance, fig_dir):
    """Original per-epoch line plot for a single experiment."""
    epochs = sorted(set(
        int(f.split('_')[-1].replace('.csv', ''))
        for f in os.listdir(exp_dir)
        if (f.startswith('in_scores_sum_') or f.startswith('in_scores_optimal_'))
        and f.endswith('.csv')
    ))
    if not epochs:
        print(f"Error: no in_scores_(sum|optimal)_*.csv files found in {exp_dir}")
        sys.exit(1)

    print(f"Found score files for epochs: {epochs}")

    epoch_list, upper_bounds, steinke_bounds, fdp_bounds, ndis_bounds = [], [], [], [], []

    for epoch in epochs:
        result = audit_epoch(exp_dir, epoch, delta, significance)
        if result is None:
            print(f"  Epoch {epoch}: missing score files, skipping")
            continue
        epoch_list.append(epoch)
        upper_bounds.append(result['upper'])
        steinke_bounds.append(result['steinke'])
        fdp_bounds.append(result['fdp'])
        ndis_bounds.append(result['ndis'])
        print(f"  Epoch {epoch:4d}: upper={result['upper']:.3f}, "
              f"steinke={result['steinke']:.3f}, fdp={result['fdp']:.3f}, ndis={result['ndis']:.3f}")

    # Save results
    results = np.column_stack([epoch_list, upper_bounds, steinke_bounds, fdp_bounds, ndis_bounds])
    results_path = os.path.join(exp_dir, 'auditing_results.csv')
    np.savetxt(results_path, results, delimiter=',',
               header='epoch,upper_bound,steinke_2023,fdp_2024,ndis', comments='')
    print(f"\nResults saved to: {results_path}")

    # Plot
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_method(ax, epoch_list, upper_bounds,   'upper')
        _plot_method(ax, epoch_list, steinke_bounds, 'steinke')
        _plot_method(ax, epoch_list, fdp_bounds,     'fdp')
        _plot_method(ax, epoch_list, ndis_bounds,    'ndis')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'$\varepsilon$')
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left')
        fig.tight_layout()
        fig_path = os.path.join(fig_dir, 'privacy_bounds_comparison.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Figure saved to: {fig_path} (and .pdf)")


def run_multi(exp_dirs, delta, significance, fig_dir):
    """Bar chart comparing final-epoch empirical eps across multiple target epsilons."""
    target_epsilons = []
    upper_bounds = []
    steinke_bounds = []
    fdp_bounds = []
    ndis_bounds = []

    for exp_dir in exp_dirs:
        if not os.path.isdir(exp_dir):
            print(f"Warning: {exp_dir} not found, skipping")
            continue

        target_eps = get_target_epsilon(exp_dir)
        if target_eps is None:
            print(f"Warning: no hparams.json in {exp_dir}, skipping")
            continue

        final_epoch = get_final_epoch(exp_dir)
        if final_epoch is None:
            print(f"Warning: no score files in {exp_dir}, skipping")
            continue

        result = audit_epoch(exp_dir, final_epoch, delta, significance)
        if result is None:
            print(f"Warning: missing score files for epoch {final_epoch} in {exp_dir}, skipping")
            continue

        target_epsilons.append(target_eps)
        upper_bounds.append(result['upper'])
        steinke_bounds.append(result['steinke'])
        fdp_bounds.append(result['fdp'])
        ndis_bounds.append(result['ndis'])

        print(f"  eps={target_eps}: epoch={final_epoch}, upper={result['upper']:.3f}, "
              f"steinke={result['steinke']:.3f}, fdp={result['fdp']:.3f}, ndis={result['ndis']:.3f}")

    if not target_epsilons:
        print("Error: no valid experiments found")
        sys.exit(1)

    # Sort by target epsilon
    order = np.argsort(target_epsilons)
    target_epsilons = [target_epsilons[i] for i in order]
    upper_bounds = [upper_bounds[i] for i in order]
    steinke_bounds = [steinke_bounds[i] for i in order]
    fdp_bounds = [fdp_bounds[i] for i in order]
    ndis_bounds = [ndis_bounds[i] for i in order]

    # Save results
    results = np.column_stack([target_epsilons, upper_bounds, steinke_bounds, fdp_bounds, ndis_bounds])
    results_path = os.path.join(fig_dir, 'auditing_comparison_final.csv')
    np.savetxt(results_path, results, delimiter=',',
               header='target_epsilon,upper_bound,steinke_2023,fdp_2024,ndis', comments='')
    print(f"\nResults saved to: {results_path}")

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(10, 6))

        _plot_method(ax, target_epsilons, upper_bounds,   'upper')
        _plot_method(ax, target_epsilons, steinke_bounds, 'steinke')
        _plot_method(ax, target_epsilons, fdp_bounds,     'fdp')
        _plot_method(ax, target_epsilons, ndis_bounds,    'ndis')

        ax.set_xlabel(r'Theoretical $\varepsilon$')
        ax.set_ylabel(r'Empirical $\varepsilon$ (lower bound)')
        ax.set_xticks(target_epsilons)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left', handlelength=2.5)
        fig.tight_layout()

        fig_path = os.path.join(fig_dir, 'privacy_bounds_comparison_multi_eps.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Figure saved to: {fig_path} (and .pdf)")


def run_complexity(exp_dir, delta, significance, fig_dir):
    """Plot empirical eps vs total canary budget (sample-complexity sweep)."""
    final_epoch = get_final_epoch(exp_dir)
    target_eps = get_target_epsilon(exp_dir)
    if final_epoch is None:
        print(f"Error: no score files in {exp_dir}")
        sys.exit(1)

    in_sum_path = _resolve_score_path(exp_dir, 'sum', 'in', final_epoch)
    out_sum_path = _resolve_score_path(exp_dir, 'sum', 'out', final_epoch)
    in_ndis_path = os.path.join(exp_dir, f'in_scores_ndis_{final_epoch:06d}.csv')
    out_ndis_path = os.path.join(exp_dir, f'out_scores_ndis_{final_epoch:06d}.csv')

    in_sum = np.loadtxt(in_sum_path, delimiter=',')
    out_sum = np.loadtxt(out_sum_path, delimiter=',')
    in_ndis = np.loadtxt(in_ndis_path, delimiter=',')
    out_ndis = np.loadtxt(out_ndis_path, delimiter=',')

    total_budgets = [100, 200, 400, 600, 1000, 1500, 2000, 3500, 5000]
    full_budget = len(in_sum) + len(out_sum)

    results = {key: [] for key in ['steinke', 'fdp', 'ndis']}
    n_trials = 50

    print(f"Running sample complexity on total budget for epoch {final_epoch} "
          f"(in={len(in_sum)}, out={len(out_sum)})...")

    for budget in total_budgets:
        trial_data = {key: [] for key in results.keys()}

        for _ in range(n_trials):
            if budget < full_budget:
                mask = np.random.rand(budget) < 0.5
                n_in = int(np.sum(mask))
                n_out = budget - n_in
                n_in = max(2, min(n_in, len(in_sum)))
                n_out = max(2, min(n_out, len(out_sum)))
            else:
                # Use the full available pool when budget meets/exceeds it.
                n_in = len(in_sum)
                n_out = len(out_sum)

            idx_in = np.random.choice(len(in_sum), n_in, replace=False)
            idx_out = np.random.choice(len(out_sum), n_out, replace=False)
            s_in, s_out = in_sum[idx_in], out_sum[idx_out]
            n_in_scores, n_out_scores = in_ndis[idx_in], out_ndis[idx_out]

            auditor = CanaryScoreAuditor(s_in, s_out)
            eps_s, _ = auditor._epsilon_one_run_all_thresholds(
                significance, delta, True, None, use_fdp=False,
            )
            eps_f, _ = auditor._epsilon_one_run_all_thresholds(
                significance, delta, True, None, use_fdp=True,
            )
            eps_n = ndis_eps_lower_bound_with_ci(
                n_in_scores, n_out_scores, delta=delta,
                alpha=significance, pool_variance=False,
            )

            trial_data['steinke'].append(eps_s)
            trial_data['fdp'].append(eps_f)
            trial_data['ndis'].append(eps_n)

        for key in results.keys():
            results[key].append(np.mean(trial_data[key]))
        print(f"  Total Budget {budget:4d} completed (avg split: {n_in}/{n_out}).")

    # Save numeric results
    arr = np.column_stack([
        total_budgets,
        results['steinke'],
        results['fdp'],
        results['ndis'],
    ])
    results_path = os.path.join(fig_dir, 'sample_complexity.csv')
    np.savetxt(results_path, arr, delimiter=',',
               header='total_budget,steinke_2023,fdp_2024,ndis', comments='')
    print(f"\nResults saved to: {results_path}")

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.grid(True, which="major", ls="-", alpha=0.2)
        ax.grid(True, which="minor", ls=":", alpha=0.1)

        if target_eps is not None:
            ax.axhline(y=target_eps, color='#555555', ls='--', lw=1.2,
                       label='Theoretical Bound')
        _plot_method(ax, total_budgets, results['steinke'], 'steinke')
        _plot_method(ax, total_budgets, results['fdp'],     'fdp')
        _plot_method(ax, total_budgets, results['ndis'],    'ndis')

        ax.set_xscale('log')
        ax.set_xticks(total_budgets)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        ax.set_xlabel('Number of Canaries ($n$)', fontweight='bold')
        ax.set_ylabel(r'Empirical $\varepsilon$ (lower bound)', fontweight='bold')
        ax.set_ylim(0, 8.5)
        ax.legend(loc='lower right', fontsize=9, frameon=True)

        plt.tight_layout()

        fig_path = os.path.join(fig_dir, 'sample_complexity_plot.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Figure saved to: {fig_path} (and .pdf)")


def main():
    parser = argparse.ArgumentParser(description='Run auditing methods comparison')
    parser.add_argument('--exp-dir', type=str, help='Single experiment directory')
    parser.add_argument('--exp-dirs', type=str, nargs='+', help='Multiple experiment directories')
    parser.add_argument('--complexity', action='store_true', help='Run sample complexity analysis')
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--significance', type=float, default=0.05)
    parser.add_argument('--fig-dir', type=str, default=None)
    args = parser.parse_args()

    if args.fig_dir is None:
        args.fig_dir = os.path.join(project_dir, 'fig')
    os.makedirs(args.fig_dir, exist_ok=True)

    if args.complexity and args.exp_dir:
        run_complexity(args.exp_dir, args.delta, args.significance, args.fig_dir)
    elif args.exp_dirs:
        run_multi(args.exp_dirs, args.delta, args.significance, args.fig_dir)
    elif args.exp_dir:
        run_single(args.exp_dir, args.delta, args.significance, args.fig_dir)
    else:
        parser.error("Provide --exp-dir or --exp-dirs. Add --complexity for sample size analysis.")


if __name__ == '__main__':
    main()