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

  # Multiple experiment directories (one per epsilon), final-epoch bar chart & ablation:
  python scripts/run_auditing_comparison.py --exp-dirs ./data/exp_eps1 ./data/exp_eps2 ./data/exp_eps4 ./data/exp_eps8

  # Sample-complexity sweep on a single experiment (Generates Main + Ablation plots):
  python scripts/run_auditing_comparison.py --complexity --exp-dir ./data/mislabeled-canaries-<seed>-5000-0.5-cifar10
"""

import sys
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from canary_score_diagnostics import within_run_orthogonality_check

_RC = {
    'font.family': 'DejaVu Sans',
    'font.size': 18,
    'axes.titlesize': 14,
    'axes.labelsize': 18,
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
    # NDIS lower-bound variants
    'ndis_parametric_bonferroni': {'color': '#02A1BA', 'marker': 'D', 'linestyle': '-', 'linewidth': 2.0, 'markersize': 6, 'zorder': 5, 'label': 'This paper'},
    'ndis_bootstrap_ellipsoid':   {'color': '#00497D', 'marker': '*', 'linestyle': '-', 'linewidth': 2.4, 'markersize': 10, 'zorder': 8, 'label': 'This paper'},
}

NDIS_METHODS = (
    'parametric_bonferroni',
    'bootstrap_ellipsoid',
    'dp_aware_bonferroni',
)
NDIS_KEYS = tuple(f'ndis_{m}' for m in NDIS_METHODS)
# Methods that should appear in plots. dp_aware_bonferroni is a diagnostic
# (drops the iid-canary assumption) and is print-only, not plotted.
NDIS_PLOT_METHODS = ('parametric_bonferroni', 'bootstrap_ellipsoid')


def _plot_method(ax, x, y, key, label_override=None):
    """Plots a method series, allowing for dynamic legend names via label_override."""
    s = _STYLE[key]
    label = label_override if label_override is not None else s['label']
    ax.plot(x, y,
            color=s['color'], marker=s['marker'], linestyle=s['linestyle'],
            linewidth=s['linewidth'], markersize=s['markersize'],
            label=label, zorder=s['zorder'],
            markerfacecolor=s['color'], markeredgecolor='white', markeredgewidth=0.6)


script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)

from auditing import CanaryScoreAuditor
from whitebox_auditing.ndis_1d import (
    ndis_eps_from_delta_1d_brentq,
    estimate_mean_variance, ndis_eps_lb, ndis_eps_lb_all,
)


def _resolve_score_path(exp_dir, kind, side, epoch):
    candidates = [
        os.path.join(exp_dir, f'{side}_scores_{kind}_{epoch:06d}.csv'),
        os.path.join(exp_dir, f'{side}_scores_optimal_{epoch:06d}.csv'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def audit_epoch(exp_dir, epoch, delta, significance, method=None, target_eps=None):
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

    emp_std = np.std(np.concatenate([in_ndis, out_ndis]), ddof=1)
    mu_sep  = np.mean(in_ndis) - np.mean(out_ndis)
    print(f"emp_std={emp_std:.4f}, mu_sep={mu_sep:.4f}")

    if os.path.isfile(privacy_path):
        pp = np.loadtxt(privacy_path, delimiter=',', skiprows=1)
        eps_upper = float(pp[0]) if pp.ndim == 1 else float(pp[0, 0])
    else:
        eps_upper = float('nan')

    # Score clip for the dp-aware bound: 1.05x the empirical max-abs so no
    # clipping bias is incurred. tau in this regime scales as sigma*sqrt(2 log n).
    tau = 1.05 * float(np.max(np.abs(np.concatenate([in_ndis, out_ndis]))))
    # eps_theory drives the worst-case posterior shift in the dp-aware bound.
    # Prefer the calibrated target from hparams; fall back to the running eps_upper.
    eps_th = target_eps if target_eps is not None else eps_upper

    auditor = CanaryScoreAuditor(in_sum, out_sum)
    eps_steinke, _ = auditor._epsilon_one_run_all_thresholds(
        significance=significance, delta=delta, one_sided=True, threshold=None, use_fdp=False,
    )
    eps_fdp, _ = auditor._epsilon_one_run_all_thresholds(
        significance=significance, delta=delta, one_sided=True, threshold=None, use_fdp=True,
    )

    ndis_results = {f'ndis_{m}': 0.0 for m in NDIS_METHODS}
    try:
        all_out = ndis_eps_lb_all(
            in_ndis, out_ndis, delta=delta,
            alpha=significance, n_bootstrap=2000,
            pool_variance=True,
            eps_theory=eps_th, score_clip=tau,
        )
        for m in NDIS_METHODS:
            if m in all_out:
                ndis_results[f'ndis_{m}'] = float(all_out[m]['eps_lb'])
    except (ValueError, RuntimeError) as e:
        print(f"  NDIS failed ({e}), setting all variants to 0")

    return {
        'upper': eps_upper,
        'steinke': eps_steinke,
        'fdp': eps_fdp,
        **ndis_results,
    }


def get_final_epoch(exp_dir):
    epochs = sorted(set(
        int(f.split('_')[-1].replace('.csv', ''))
        for f in os.listdir(exp_dir)
        if (f.startswith('in_scores_sum_') or f.startswith('in_scores_optimal_'))
        and f.endswith('.csv')
    ))
    return epochs[-1] if epochs else None


def get_target_epsilon(exp_dir):
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

    target_eps = get_target_epsilon(exp_dir)
    series = {k: [] for k in ('epoch', 'upper', 'steinke', 'fdp', *NDIS_KEYS)}

    for epoch in epochs:
        result = audit_epoch(exp_dir, epoch, delta, significance, target_eps=target_eps)
        if result is None:
            print(f"  Epoch {epoch}: missing score files, skipping")
            continue
        series['epoch'].append(epoch)
        for k in ('upper', 'steinke', 'fdp', *NDIS_KEYS):
            series[k].append(result[k])
        ndis_str = ', '.join(f"{k.replace('ndis_', '')[:14]}={result[k]:.3f}" for k in NDIS_KEYS)
        print(f"  Epoch {epoch:4d}: upper={result['upper']:.3f}, "
              f"steinke={result['steinke']:.3f}, fdp={result['fdp']:.3f}, {ndis_str}")

    cols = ('epoch', 'upper', 'steinke', 'fdp', *NDIS_KEYS)
    results = np.column_stack([series[c] for c in cols])
    results_path = os.path.join(exp_dir, 'auditing_results.csv')
    np.savetxt(results_path, results, delimiter=',',
               header=','.join(cols), comments='')
    print(f"\nResults saved to: {results_path}")

    # Plot (Only plotting Ellipsoid for the NDIS method to keep it clean)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_method(ax, series['epoch'], series['upper'],   'upper')
        _plot_method(ax, series['epoch'], series['steinke'], 'steinke')
        _plot_method(ax, series['epoch'], series['fdp'],     'fdp')
        _plot_method(ax, series['epoch'], series['ndis_bootstrap_ellipsoid'], 'ndis_bootstrap_ellipsoid')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'$\varepsilon$')
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left', fontsize=14)
        fig.tight_layout()

        fig_path = os.path.join(fig_dir, 'privacy_bounds_comparison.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Figure saved to: {fig_path} (and .pdf)")


def run_multi(exp_dirs, delta, significance, fig_dir):
    """Compare final-epoch empirical eps across multiple target epsilons."""
    series = {k: [] for k in ('target', 'upper', 'steinke', 'fdp', *NDIS_KEYS)}
    ortho_results = []
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

        result = audit_epoch(exp_dir, final_epoch, delta, significance, target_eps=target_eps)
        if result is None:
            print(f"Warning: missing score files for epoch {final_epoch} in {exp_dir}, skipping")
            continue

        series['target'].append(target_eps)
        for k in ('upper', 'steinke', 'fdp', *NDIS_KEYS):
            series[k].append(result[k])

        dirs_path = os.path.join(exp_dir, 'canary_directions.csv')
        if os.path.isfile(dirs_path):
            indices = np.loadtxt(dirs_path, delimiter=',', skiprows=1)
            m = len(indices)
            num_unique = len(np.unique(indices, axis=0))
            collisions = m - num_unique
            ortho_results.append({
                'target': target_eps,
                'm': m,
                'collisions': collisions,
                'orthogonal': collisions == 0
            })
        else:
            ortho_results.append(None)

        ndis_str = ', '.join(f"{k.replace('ndis_', '')[:14]}={result[k]:.3f}" for k in NDIS_KEYS)
        print(f"  eps={target_eps}: epoch={final_epoch}, upper={result['upper']:.3f}, "
              f"steinke={result['steinke']:.3f}, fdp={result['fdp']:.3f}, {ndis_str}")

    if not series['target']:
        print("Error: no valid experiments found")
        sys.exit(1)

    order = np.argsort(series['target'])
    for k in series:
        series[k] = [series[k][i] for i in order]

    cols = ('target', 'upper', 'steinke', 'fdp', *NDIS_KEYS)
    results = np.column_stack([series[c] for c in cols])
    results_path = os.path.join(fig_dir, 'auditing_comparison_final.csv')
    np.savetxt(results_path, results, delimiter=',',
               header=','.join(cols), comments='')
    print(f"\nResults saved to: {results_path}")


    if any(ortho_results):
        print("\n" + "="*60)
        print("CANARY INDEPENDENCE DIAGNOSTIC (DIRAC ORTHOGONALITY)")
        print("-" * 60)
        print(f"{'Target Eps':>10} | {'Canaries (m)':>12} | {'Collisions':>10} | {'Status'}")
        print("-" * 60)
        for res in ortho_results:
            if res:
                status = "PASS (IID)" if res['orthogonal'] else "FAIL (CORRELATED)"
                print(f"{res['target']:10.2f} | {res['m']:12d} | {res['collisions']:10d} | {status}")
        print("="*60)
        print("Note: Dirac canaries are perfectly orthogonal if collisions = 0.")

    # ==========================================
    # 1. Main Plot: Comparison with all methods (Ellipsoid only for NDIS)
    # ==========================================
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        _plot_method(ax, series['target'], series['upper'],   'upper')
        _plot_method(ax, series['target'], series['steinke'], 'steinke')
        _plot_method(ax, series['target'], series['fdp'],     'fdp')
        _plot_method(ax, series['target'], series['ndis_bootstrap_ellipsoid'], 'ndis_bootstrap_ellipsoid')

        ax.set_xlabel(r'Theoretical $\varepsilon$')
        ax.set_ylabel(r'Empirical $\varepsilon$ (lower bound)')
        ax.set_xticks(series['target'])
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left', handlelength=2.5, fontsize=14)
        fig.tight_layout()

        fig_path = os.path.join(fig_dir, 'privacy_bounds_comparison_multi_eps.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Main figure saved to: {fig_path} (and .pdf)")

    # ==========================================
    # 2. Ablation Plot: CR Geometry Comparison
    # ==========================================
    with plt.rc_context(_RC):
        fig_abl, ax_abl = plt.subplots(figsize=(11, 6.5))
        _plot_method(ax_abl, series['target'], series['upper'], 'upper')

        _plot_method(ax_abl, series['target'], series['ndis_parametric_bonferroni'], 'ndis_parametric_bonferroni')
        # Explicit override for the ablation plot only
        _plot_method(ax_abl, series['target'], series['ndis_bootstrap_ellipsoid'], 'ndis_bootstrap_ellipsoid', label_override='This paper (Bootstrap Ellipsoid)')

        ax_abl.set_xlabel(r'Theoretical $\varepsilon$')
        ax_abl.set_ylabel(r'Empirical $\varepsilon$ (lower bound)')
        ax_abl.set_xticks(series['target'])
        ax_abl.set_ylim(bottom=0)
        ax_abl.legend(loc='upper left', handlelength=2.5, fontsize=14)
        fig_abl.tight_layout()

        ablation_fig_path = os.path.join(fig_dir, 'ablation_cr_geometry_multi_eps.png')
        fig_abl.savefig(ablation_fig_path, dpi=300, bbox_inches='tight')
        fig_abl.savefig(ablation_fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Ablation figure saved to: {ablation_fig_path} (and .pdf)")


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

    total_budgets = [50, 100, 200, 400, 600, 1000, 1500, 2000, 3500, 5000]
    full_budget = len(in_sum) + len(out_sum)

    metric_keys = ('steinke', 'fdp', *NDIS_KEYS)
    results = {k: [] for k in metric_keys}
    n_trials = 50

    print(f"Running sample complexity on total budget for epoch {final_epoch} "
          f"(in={len(in_sum)}, out={len(out_sum)})...")

    for budget in total_budgets:
        trial_data = {k: [] for k in metric_keys}

        for _ in range(n_trials):
            if budget < full_budget:
                mask = np.random.rand(budget) < 0.5
                n_in = int(np.sum(mask))
                n_out = budget - n_in
                n_in = max(2, min(n_in, len(in_sum)))
                n_out = max(2, min(n_out, len(out_sum)))
            else:
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
            tau = 1.05 * float(np.max(np.abs(np.concatenate([n_in_scores, n_out_scores]))))
            try:
                ndis_out = ndis_eps_lb_all(
                    n_in_scores, n_out_scores, delta=delta,
                    alpha=significance, pool_variance=True,
                    eps_theory=target_eps, score_clip=tau,
                )
            except (ValueError, RuntimeError):
                ndis_out = None

            trial_data['steinke'].append(eps_s)
            trial_data['fdp'].append(eps_f)
            for m in NDIS_METHODS:
                v = float(ndis_out[m]['eps_lb']) if (ndis_out is not None and m in ndis_out) else 0.0
                trial_data[f'ndis_{m}'].append(v)

        for k in metric_keys:
            results[k].append(np.mean(trial_data[k]))
        print(f"  Total Budget {budget:4d} completed (avg split: {n_in}/{n_out}).")

    cols = ('total_budget', *metric_keys)
    arr = np.column_stack([total_budgets, *[results[k] for k in metric_keys]])
    results_path = os.path.join(fig_dir, 'sample_complexity.csv')
    np.savetxt(results_path, arr, delimiter=',',
               header=','.join(cols), comments='')
    print(f"\nResults saved to: {results_path}")

    # ==========================================
    # 1. Main Plot: Competitors vs NDIS Ellipsoid
    # ==========================================
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.grid(True, which="major", ls="-", alpha=0.2)
        ax.grid(True, which="minor", ls=":", alpha=0.1)

        if target_eps is not None:
            ax.axhline(y=target_eps, color='#555555', ls='--', lw=1.2,
                       label='Theoretical Upper Bound')
        _plot_method(ax, total_budgets, results['steinke'], 'steinke')
        _plot_method(ax, total_budgets, results['fdp'],     'fdp')
        _plot_method(ax, total_budgets, results['ndis_bootstrap_ellipsoid'], 'ndis_bootstrap_ellipsoid')

        ax.set_xscale('log')
        ax.set_xticks(total_budgets)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        ax.set_xlabel('Number of Canaries ($n$)')
        ax.set_ylabel(r'Empirical $\varepsilon$ (lower bound)')
        ax.set_ylim(0, 8.5)
        ax.legend(loc='lower right', fontsize=14, frameon=True)
        plt.tight_layout()

        fig_path = os.path.join(fig_dir, 'sample_complexity_main.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Main Sample Complexity Figure saved to: {fig_path} (and .pdf)")

    # ==========================================
    # 2. Ablation Plot: CR Geometry Comparison
    # ==========================================
    with plt.rc_context(_RC):
        fig_abl, ax_abl = plt.subplots(figsize=(9, 5.5))
        ax_abl.grid(True, which="major", ls="-", alpha=0.2)
        ax_abl.grid(True, which="minor", ls=":", alpha=0.1)

        if target_eps is not None:
            ax_abl.axhline(y=target_eps, color='#555555', ls='--', lw=1.2,
                           label='Theoretical Upper Bound')

        _plot_method(ax_abl, total_budgets, results['ndis_parametric_bonferroni'], 'ndis_parametric_bonferroni')
        # Explicit override for the ablation plot only
        _plot_method(ax_abl, total_budgets, results['ndis_bootstrap_ellipsoid'], 'ndis_bootstrap_ellipsoid', label_override='This paper (Bootstrap Ellipsoid)')

        ax_abl.set_xscale('log')
        ax_abl.set_xticks(total_budgets)
        ax_abl.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        ax_abl.set_xlabel('Number of Canaries ($n$)', fontweight='bold')
        ax_abl.set_ylabel(r'Empirical $\varepsilon$ (lower bound)', fontweight='bold')
        ax_abl.set_ylim(0, 8.5)
        ax_abl.legend(loc='lower right', fontsize=14, frameon=True)
        plt.tight_layout()

        ablation_fig_path = os.path.join(fig_dir, 'ablation_cr_sample_complexity.png')
        plt.savefig(ablation_fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(ablation_fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Ablation Sample Complexity Figure saved to: {ablation_fig_path} (and .pdf)")


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