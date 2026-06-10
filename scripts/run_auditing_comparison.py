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
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from canary_score_diagnostics import within_run_orthogonality_check

_RC = {
    'font.family': 'DejaVu Sans',
    'font.size': 22,
    'axes.titlesize': 22,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
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
    'legend.frameon': True,
    'legend.fontsize': 16,
    'legend.loc': 'upper left',
    'figure.dpi': 120,
}

_STYLE = {
    'upper':   {'color': '#555555', 'marker': '',  'linestyle': (0, (3, 5, 1, 5)),  'linewidth': 1.4, 'markersize': 0,  'zorder': 1, 'label': 'Theoretical Upper Bound)'},
    'steinke': {'color': '#ff7f0e', 'marker': 'o', 'linestyle': '--',  'linewidth': 2.4, 'markersize': 7,  'zorder': 2, 'label': 'Steinke et al. 2023'},
    'fdp':     {'color': '#2ca02c', 'marker': 's', 'linestyle': '--',  'linewidth': 2.4, 'markersize': 7,  'zorder': 3, 'label': 'Mahloujifar et al. 2024 (f-DP)'},
    'andrew':  {'color': '#9467bd', 'marker': '^', 'linestyle': '--',  'linewidth': 2.4, 'markersize': 8,  'zorder': 4, 'label': 'Andrew et al. 2024'},
    # NDIS lower-bound variants
    'ndis_parametric_bonferroni': {'color': '#02A1BA', 'marker': 'D', 'linestyle': '-', 'linewidth': 2.0, 'markersize': 6, 'zorder': 5, 'label': 'This paper'},
    'ndis_bootstrap_ellipsoid':   {'color': '#00497D', 'marker': '*', 'linestyle': '-', 'linewidth': 2.4, 'markersize': 10, 'zorder': 8, 'label': 'This paper'},
}

NDIS_METHODS = (
    'parametric_bonferroni',
    'bootstrap_ellipsoid',
)
NDIS_KEYS = tuple(f'ndis_{m}' for m in NDIS_METHODS)
NDIS_PLOT_METHODS = NDIS_METHODS


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
    _ndis_eps_from_moments,
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


def audit_epoch(exp_dir, epoch, delta, significance, method=None, target_eps=None,
                with_andrew=False):
    in_sum_path = _resolve_score_path(exp_dir, 'sum', 'in', epoch)
    out_sum_path = _resolve_score_path(exp_dir, 'sum', 'out', epoch)
    in_ndis_path = os.path.join(exp_dir, f'in_scores_ndis_{epoch:06d}.csv')
    out_ndis_path = os.path.join(exp_dir, f'out_scores_ndis_{epoch:06d}.csv')
    in_andrew_path = os.path.join(exp_dir, f'in_scores_andrew_{epoch:06d}.csv')
    out_andrew_path = os.path.join(exp_dir, f'out_scores_andrew_{epoch:06d}.csv')
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
        )
        for m in NDIS_METHODS:
            if m in all_out:
                ndis_results[f'ndis_{m}'] = float(all_out[m]['eps_lb'])
    except (ValueError, RuntimeError) as e:
        print(f"  NDIS failed ({e}), setting all variants to 0")

    out = {
        'upper': eps_upper,
        'steinke': eps_steinke,
        'fdp': eps_fdp,
        **ndis_results,
    }

    if with_andrew:
        eps_andrew = 0.0
        if os.path.isfile(in_andrew_path) and os.path.isfile(out_andrew_path):
            in_andrew = np.loadtxt(in_andrew_path, delimiter=',')
            out_andrew = np.loadtxt(out_andrew_path, delimiter=',')
            if in_andrew.size >= 2 and out_andrew.size >= 2:
                eps_andrew = float(_ndis_eps_from_moments(
                    in_mean=float(np.mean(in_andrew)),
                    in_std=float(np.std(in_andrew, ddof=1)),
                    out_mean=float(np.mean(out_andrew)),
                    out_std=float(np.std(out_andrew, ddof=1)),
                    delta=delta,
                    pool_variance=False,
                ))
        out['andrew'] = eps_andrew

    return out


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


def get_target_T(exp_dir):
    hparams_path = os.path.join(exp_dir, 'hparams.json')
    if os.path.isfile(hparams_path):
        with open(hparams_path) as f:
            return json.load(f).get('target_steps')
    return None


def run_single(exp_dir, delta, significance, fig_dir, with_andrew=False):
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
    metric_keys = ('upper', 'steinke', 'fdp', *(('andrew',) if with_andrew else ()), *NDIS_KEYS)
    series = {k: [] for k in ('epoch', *metric_keys)}

    for epoch in epochs:
        result = audit_epoch(
            exp_dir, epoch, delta, significance, target_eps=target_eps,
            with_andrew=with_andrew,
        )
        if result is None:
            print(f"  Epoch {epoch}: missing score files, skipping")
            continue
        series['epoch'].append(epoch)
        for k in metric_keys:
            series[k].append(result[k])
        ndis_str = ', '.join(f"{k.replace('ndis_', '')[:14]}={result[k]:.3f}" for k in NDIS_KEYS)
        andrew_str = f"andrew={result['andrew']:.3f}, " if with_andrew else ''
        print(f"  Epoch {epoch:4d}: upper={result['upper']:.3f}, "
              f"steinke={result['steinke']:.3f}, fdp={result['fdp']:.3f}, "
              f"{andrew_str}{ndis_str}")

    cols = ('epoch', *metric_keys)
    results = np.column_stack([series[c] for c in cols])
    results_path = os.path.join(exp_dir, 'auditing_results.csv')
    np.savetxt(results_path, results, delimiter=',',
               header=','.join(cols), comments='')
    print(f"\nResults saved to: {results_path}")

    # Plot (Only plotting Ellipsoid for the NDIS method to keep it clean)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_method(ax, series['epoch'], series['upper'],   'upper')
        if with_andrew:
            # DP-FTRL panel: ours vs Andrew only. Steinke / Mahloujifar are
            # designed for DP-SGD-style score distributions and aren't the
            # right baseline once we move to the tree-mechanism setting.
            _plot_method(ax, series['epoch'], series['andrew'], 'andrew')
        else:
            _plot_method(ax, series['epoch'], series['steinke'], 'steinke')
            _plot_method(ax, series['epoch'], series['fdp'],     'fdp')
        _plot_method(ax, series['epoch'], series['ndis_bootstrap_ellipsoid'], 'ndis_bootstrap_ellipsoid')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'$\varepsilon$')
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left')
        fig.tight_layout()

        fig_path = os.path.join(fig_dir, 'privacy_bounds_comparison.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Figure saved to: {fig_path} (and .pdf)")


def run_multi(exp_dirs, delta, significance, fig_dir, with_andrew=False):
    """Compare final-epoch empirical eps across multiple target epsilons."""
    metric_keys = ('upper', 'steinke', 'fdp', *(('andrew',) if with_andrew else ()), *NDIS_KEYS)
    series = {k: [] for k in ('target', *metric_keys)}
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

        result = audit_epoch(
            exp_dir, final_epoch, delta, significance, target_eps=target_eps,
            with_andrew=with_andrew,
        )
        if result is None:
            print(f"Warning: missing score files for epoch {final_epoch} in {exp_dir}, skipping")
            continue

        series['target'].append(target_eps)
        for k in metric_keys:
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
        andrew_str = f"andrew={result['andrew']:.3f}, " if with_andrew else ''
        print(f"  eps={target_eps}: epoch={final_epoch}, upper={result['upper']:.3f}, "
              f"steinke={result['steinke']:.3f}, fdp={result['fdp']:.3f}, "
              f"{andrew_str}{ndis_str}")

    if not series['target']:
        print("Error: no valid experiments found")
        sys.exit(1)

    order = np.argsort(series['target'])
    for k in series:
        series[k] = [series[k][i] for i in order]

    cols = ('target', *metric_keys)
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

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        _plot_method(ax, series['target'], series['upper'],   'upper')
        if with_andrew:
            _plot_method(ax, series['target'], series['andrew'], 'andrew')
        else:
            _plot_method(ax, series['target'], series['steinke'], 'steinke')
            _plot_method(ax, series['target'], series['fdp'],     'fdp')
        _plot_method(ax, series['target'], series['ndis_bootstrap_ellipsoid'], 'ndis_bootstrap_ellipsoid')

        ax.set_xlabel(r'Theoretical $\varepsilon$')
        ax.set_ylabel(r'Empirical $\varepsilon$ (lower bound)')
        ax.set_xticks(series['target'])
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left', handlelength=2.5)
        fig.tight_layout()

        fig_path = os.path.join(fig_dir, 'privacy_bounds_comparison_multi_eps.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Main figure saved to: {fig_path} (and .pdf)")

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
        ax_abl.legend(loc='upper left', handlelength=2.5)
        fig_abl.tight_layout()

        ablation_fig_path = os.path.join(fig_dir, 'ablation_cr_geometry_multi_eps.png')
        fig_abl.savefig(ablation_fig_path, dpi=300, bbox_inches='tight')
        fig_abl.savefig(ablation_fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Ablation figure saved to: {ablation_fig_path} (and .pdf)")


def run_ablation_T(exp_dirs, delta, significance, fig_dir, with_andrew=False):
    """Empirical eps vs number of training steps T at fixed target epsilon.
    """
    metric_keys = ('upper', 'steinke', 'fdp', *(('andrew',) if with_andrew else ()), *NDIS_KEYS)
    series = {k: [] for k in ('T', *metric_keys)}
    target_eps_seen = []
    for exp_dir in exp_dirs:
        if not os.path.isdir(exp_dir):
            print(f"Warning: {exp_dir} not found, skipping")
            continue
        target_eps = get_target_epsilon(exp_dir)
        T = get_target_T(exp_dir)
        if target_eps is None or T is None:
            print(f"Warning: hparams.json missing epsilon or target_steps in {exp_dir}, skipping")
            continue
        final_epoch = get_final_epoch(exp_dir)
        if final_epoch is None:
            print(f"Warning: no score files in {exp_dir}, skipping")
            continue
        result = audit_epoch(
            exp_dir, final_epoch, delta, significance, target_eps=target_eps,
            with_andrew=with_andrew,
        )
        if result is None:
            print(f"Warning: missing score files for epoch {final_epoch} in {exp_dir}, skipping")
            continue

        target_eps_seen.append(float(target_eps))
        series['T'].append(int(T))
        for k in metric_keys:
            series[k].append(result[k])

        ndis_str = ', '.join(f"{k.replace('ndis_', '')[:14]}={result[k]:.3f}" for k in NDIS_KEYS)
        andrew_str = f"andrew={result['andrew']:.3f}, " if with_andrew else ''
        print(f"  T={T}: eps_target={target_eps}, epoch={final_epoch}, "
              f"upper={result['upper']:.3f}, steinke={result['steinke']:.3f}, "
              f"fdp={result['fdp']:.3f}, {andrew_str}{ndis_str}")

    if not series['T']:
        print("Error: no valid experiments found")
        sys.exit(1)

    eps_set = sorted({round(e, 4) for e in target_eps_seen})
    if len(eps_set) > 1:
        print(f"Warning: target eps differs across exp_dirs ({eps_set}); using first run's eps "
              "for the reference line.")
    target_eps_label = target_eps_seen[0]

    order = np.argsort(series['T'])
    for k in series:
        series[k] = [series[k][i] for i in order]

    cols = ('T', *metric_keys)
    results = np.column_stack([series[c] for c in cols])
    eps_tag = f"{target_eps_label:g}".replace('.', 'p')
    results_path = os.path.join(fig_dir, f'auditing_ablation_T_eps{eps_tag}.csv')
    np.savetxt(results_path, results, delimiter=',', header=','.join(cols), comments='')
    print(f"\nResults saved to: {results_path}")

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.axhline(y=target_eps_label, color='#555555', ls=(0, (3, 5, 1, 5)), lw=1.4,
                   label=fr'Theoretical $\varepsilon = {target_eps_label:g}$', zorder=1)
        if with_andrew:
            _plot_method(ax, series['T'], series['andrew'], 'andrew')
        else:
            _plot_method(ax, series['T'], series['steinke'], 'steinke')
            _plot_method(ax, series['T'], series['fdp'],     'fdp')
        _plot_method(ax, series['T'], series['ndis_bootstrap_ellipsoid'], 'ndis_bootstrap_ellipsoid')

        ax.set_xscale('log')
        ax.set_xticks(series['T'])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel(r'Number of training steps $T$')
        ax.set_ylabel(r'Empirical $\varepsilon$ (lower bound)')
        ax.set_ylim(bottom=0)
        ax.legend()
        fig.tight_layout()

        fig_path = os.path.join(fig_dir, f'ablation_T_eps{eps_tag}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Main figure saved to: {fig_path} (and .pdf)")

    with plt.rc_context(_RC):
        fig_abl, ax_abl = plt.subplots(figsize=(11, 6.5))
        ax_abl.axhline(y=target_eps_label, color='#555555', ls=(0, (3, 5, 1, 5)), lw=1.4,
                       label=fr'Theoretical $\varepsilon = {target_eps_label:g}$', zorder=1)
        _plot_method(ax_abl, series['T'], series['ndis_parametric_bonferroni'], 'ndis_parametric_bonferroni')
        _plot_method(ax_abl, series['T'], series['ndis_bootstrap_ellipsoid'], 'ndis_bootstrap_ellipsoid',
                     label_override='This paper (Bootstrap Ellipsoid)')

        ax_abl.set_xscale('log')
        ax_abl.set_xticks(series['T'])
        ax_abl.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax_abl.set_xlabel(r'Number of training steps $T$')
        ax_abl.set_ylabel(r'Empirical $\varepsilon$ (lower bound)')
        ax_abl.set_ylim(bottom=0)
        ax_abl.legend()
        fig_abl.tight_layout()

        ablation_fig_path = os.path.join(fig_dir, f'ablation_cr_geometry_T_eps{eps_tag}.png')
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
        # Sparse, log-friendly major ticks so 1000/1500/2000/3500/5000 don't
        # crash into each other; the data markers still sit at every
        # `total_budgets` entry because `_plot_method` uses them as x.
        major_ticks = [t for t in (50, 100, 200, 500, 1000, 2000, 5000)
                       if min(total_budgets) <= t <= max(total_budgets)]
        ax.set_xticks(major_ticks)
        ax.set_xticks(total_budgets, minor=True)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(axis='x', which='minor', labelbottom=False)

        ax.set_xlabel('Number of Canaries ($m$)')
        ax.set_ylabel(r'Empirical $\varepsilon$ (lower bound)')
        ax.set_ylim(0, 8.5)
        ax.legend()
        plt.tight_layout()

        fig_path = os.path.join(fig_dir, 'sample_complexity_main.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Main Sample Complexity Figure saved to: {fig_path} (and .pdf)")

    with plt.rc_context(_RC):
        fig_abl, ax_abl = plt.subplots(figsize=(9, 5.5))
        ax_abl.grid(True, which="major", ls="-", alpha=0.2)
        ax_abl.grid(True, which="minor", ls=":", alpha=0.1)

        if target_eps is not None:
            ax_abl.axhline(y=target_eps, color='#555555', ls='--', lw=1.2,
                           label='Theoretical Upper Bound')

        _plot_method(ax_abl, total_budgets, results['ndis_parametric_bonferroni'], 'ndis_parametric_bonferroni',label_override='95% Bonferroni confidence')
        # Explicit override for the ablation plot only
        _plot_method(ax_abl, total_budgets, results['ndis_bootstrap_ellipsoid'], 'ndis_bootstrap_ellipsoid', label_override='95% Bootstrap Ellipsoid confidence')

        ax_abl.set_xscale('log')
        major_ticks = [t for t in (50, 100, 200, 500, 1000, 2000, 5000)
                       if min(total_budgets) <= t <= max(total_budgets)]
        ax_abl.set_xticks(major_ticks)
        ax_abl.set_xticks(total_budgets, minor=True)
        ax_abl.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax_abl.tick_params(axis='x', which='minor', labelbottom=False)

        ax_abl.set_xlabel('Number of Canaries ($m$)')
        ax_abl.set_ylabel(r'Empirical $\varepsilon$')
        ax_abl.set_ylim(0, 8.5)
        ax_abl.legend()
        plt.tight_layout()

        ablation_fig_path = os.path.join(fig_dir, 'ablation_cr_sample_complexity.png')
        plt.savefig(ablation_fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(ablation_fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Ablation Sample Complexity Figure saved to: {ablation_fig_path} (and .pdf)")


def run_independence(exp_dir, fig_dir, *, m_show=200, with_andrew=False):
    """Visual canary-independence diagnostic from a single exp_dir.

    Two panels (shared 'Blues' palette):
      (i)  Gram matrix |<e_{c_i}, e_{c_j}>| for the first m_show canaries.
           For dirac canaries with distinct coords this is the identity --
           visual confirmation of mutual orthogonality, the necessary
           condition for the iid per-canary-score model used by NDIS.
      (ii) Hexbin density of (canary index, standardised score) for in-
           and out-canaries side by side. If scores are exchangeable across
           canary index, the density is uniform along the x-axis with a
           Gaussian profile in z.
    """
    if not os.path.isdir(exp_dir):
        sys.exit(f"exp_dir not found: {exp_dir}")

    final_epoch = get_final_epoch(exp_dir)
    if final_epoch is None:
        sys.exit(f"No score files in {exp_dir}")

    coords_path = None
    for cand in ('canary_coords.csv', 'canary_directions.csv'):
        p = os.path.join(exp_dir, cand)
        if os.path.isfile(p):
            coords_path = p
            break
    if coords_path is None:
        sys.exit(f"No canary_coords.csv / canary_directions.csv in {exp_dir}")
    coords = np.loadtxt(coords_path, delimiter=',', skiprows=1).astype(int)
    if coords.ndim == 1:
        coords = coords[:, None]

    in_path = _resolve_score_path(exp_dir, 'sum', 'in', final_epoch)
    out_path = _resolve_score_path(exp_dir, 'sum', 'out', final_epoch)
    if in_path is None or out_path is None:
        sys.exit(f"No score files for epoch {final_epoch} in {exp_dir}")
    in_scores = np.loadtxt(in_path, delimiter=',')
    out_scores = np.loadtxt(out_path, delimiter=',')

    inc_mask_path = os.path.join(exp_dir, 'inclusion_mask.csv')
    inclusion_mask = np.loadtxt(inc_mask_path, delimiter=',').astype(bool) \
        if os.path.isfile(inc_mask_path) else None

    m_total = int(coords.shape[0])
    m_show = int(min(m_show, m_total))
    sub = coords[:m_show]
    if sub.shape[1] >= 2:
        same_p = sub[:, 0:1] == sub[:, 0:1].T
        same_f = sub[:, 1:2] == sub[:, 1:2].T
        gram = (same_p & same_f).astype(float)
    else:
        gram = (sub[:, 0:1] == sub[:, 0:1].T).astype(float)

    diag_mask = np.eye(m_show, dtype=bool)
    off = gram[~diag_mask]
    print(
        f"Gram (m_show={m_show}): diag={float(gram[diag_mask].mean()):.3f}, "
        f"max|off-diag|={float(np.max(np.abs(off))):.3e}, "
        f"mean|off-diag|={float(np.mean(np.abs(off))):.3e}"
    )

    in_z = (in_scores - in_scores.mean()) / max(in_scores.std(ddof=1), 1e-12)
    out_z = (out_scores - out_scores.mean()) / max(out_scores.std(ddof=1), 1e-12)
    if inclusion_mask is not None and inclusion_mask.shape[0] == m_total:
        in_idx = np.flatnonzero(inclusion_mask)
        out_idx = np.flatnonzero(~inclusion_mask)
    else:
        # Fall back to score-aligned indices.
        in_idx = np.arange(len(in_scores))
        out_idx = np.arange(len(out_scores))

    target_eps = get_target_epsilon(exp_dir)
    eps_tag = f"{target_eps:g}".replace('.', 'p') if target_eps is not None else 'unknown'

    base = os.path.join(fig_dir, f'canary_independence_eps{eps_tag}')
    with plt.rc_context(_RC):
        # ---- Plot 1: Gram matrix heatmap ----------------------------------
        fig_g, ax_g = plt.subplots(figsize=(8.5, 7.5))
        im = ax_g.imshow(
            gram, cmap='Blues', vmin=0.0, vmax=1.0, aspect='equal',
            interpolation='nearest',
        )
        ax_g.set_xlabel(r'Canary index $j$')
        ax_g.set_ylabel(r'Canary index $i$')
        cb = fig_g.colorbar(im, ax=ax_g, shrink=0.85, pad=0.02)
        cb.set_label(r'$|\langle e_{c_i}, e_{c_j}\rangle|$')
        fig_g.tight_layout()
        fig_g.savefig(f'{base}_gram.png', dpi=300, bbox_inches='tight')
        fig_g.savefig(f'{base}_gram.pdf',                bbox_inches='tight')

        # ---- Plot 2: Hexbin density of (canary index, standardised z) -----
        all_idx = np.concatenate([in_idx[:len(in_z)], out_idx[:len(out_z)]])
        all_z   = np.concatenate([in_z,             out_z])
        fig_d, ax_d = plt.subplots(figsize=(11, 6.5))
        hb = ax_d.hexbin(
            all_idx, all_z,
            gridsize=(60, 36), cmap='Blues', mincnt=1, linewidths=0,
        )
        ax_d.set_xlabel('Canary index')
        ax_d.set_ylabel(r'Standardised score $z$')
        ax_d.set_xlim(0, max(in_idx.max(), out_idx.max()) + 1)
        ax_d.set_ylim(-4.5, 4.5)
        cb2 = fig_d.colorbar(hb, ax=ax_d, shrink=0.85, pad=0.02)
        cb2.set_label('count per hex')
        fig_d.tight_layout()
        fig_d.savefig(f'{base}_density.png', dpi=300, bbox_inches='tight')
        fig_d.savefig(f'{base}_density.pdf',                bbox_inches='tight')

    print(f"Saved canary independence diagnostics to: {base}_{{gram,density}}.{{png,pdf}}")


def run_gaussianity(exp_dir, fig_dir):
    """Histogram of canary scores against the asymptotic Gaussian.

    Two side-by-side panels share an analytic mean / variance pinned by the
    mechanism's noise calibration (NDIS scale, score = sum / sqrt(T) or
    optimal / sqrt(L)):

      DP-FTRL:
          OUT ~ N(0,            sigma_node^2)
          IN  ~ N(sqrt(L) * C,  sigma_node^2)         with L = log2(T) + 1
      DP-SGD (Model 1, Bernoulli per-step injection at rate q = B/N):
          OUT ~ N(0,                  (sigma_dpsgd * C)^2)
          IN  ~ N(q * sqrt(T) * C,    (sigma_dpsgd * C)^2)

    The IN-mean is exactly the "qC-style" Gaussian: q * sqrt(T) * C in
    NDIS scale (or q*T*C in raw-sum scale). Comparing the empirical
    histogram to that curve is the visual Gaussianity check.
    """
    if not os.path.isdir(exp_dir):
        sys.exit(f"exp_dir not found: {exp_dir}")
    final_epoch = get_final_epoch(exp_dir)
    if final_epoch is None:
        sys.exit(f"No score files in {exp_dir}")

    in_path  = os.path.join(exp_dir, f'in_scores_ndis_{final_epoch:06d}.csv')
    out_path = os.path.join(exp_dir, f'out_scores_ndis_{final_epoch:06d}.csv')
    if not (os.path.isfile(in_path) and os.path.isfile(out_path)):
        sys.exit(f"Missing in_scores_ndis / out_scores_ndis at epoch {final_epoch}")
    in_scores  = np.loadtxt(in_path,  delimiter=',')
    out_scores = np.loadtxt(out_path, delimiter=',')

    hp_path = os.path.join(exp_dir, 'hparams.json')
    if not os.path.isfile(hp_path):
        sys.exit(f"Missing {hp_path}")
    with open(hp_path) as f:
        hp = json.load(f)

    if 'sigma_node' in hp:
        # DP-FTRL: deterministic single-leaf injection -> sqrt(L) * C mean.
        from whitebox_auditing.tree_mechanism import num_levels
        sigma_pred = float(hp['sigma_node'])
        T = int(hp['target_steps'])
        C = float(hp.get('max_grad_norm', 1.0))
        L = num_levels(T) + 1
        mean_in_pred = math.sqrt(L) * C
        mech_label = 'DP-FTRL'
        in_label = fr'$\sqrt{{L}}\cdot C = \sqrt{{{L}}}\cdot {C:g}$'
    elif 'noise_multiplier' in hp:
        # DP-SGD Model 1: per-step Bernoulli(q) injection -> q * sqrt(T) * C mean.
        nm = float(hp['noise_multiplier'])
        C = float(hp.get('max_grad_norm', 1.0))
        sigma_pred = nm * C
        T = int(hp['target_steps'])
        B = int(hp.get('logical_batch_size', 4096))
        N = int(hp.get('train_set_size', 50000))   # CIFAR-10 default
        q = B / N
        mean_in_pred = q * math.sqrt(T) * C
        mech_label = 'DP-SGD'
        in_label = fr'$qC\sqrt{{T}} = {q:.3g}\cdot {C:g}\sqrt{{{T}}}$'
    else:
        sys.exit("hparams.json has neither sigma_node nor noise_multiplier")

    target_eps = hp.get('epsilon', None)
    eps_tag = f"{float(target_eps):g}".replace('.', 'p') if target_eps is not None else 'unknown'

    out_mu, out_sd = float(out_scores.mean()), float(out_scores.std(ddof=1))
    in_mu,  in_sd  = float(in_scores.mean()),  float(in_scores.std(ddof=1))
    print(
        f"{mech_label} Gaussianity:\n"
        f"  OUT predicted N(0, sigma={sigma_pred:.4f}); empirical (mu={out_mu:.4f}, sigma={out_sd:.4f})\n"
        f"  IN  predicted N(mu={mean_in_pred:.4f}, sigma={sigma_pred:.4f});"
        f" empirical (mu={in_mu:.4f}, sigma={in_sd:.4f})"
    )

    def _hist_with_pdf(ax, scores, mean_pred, sigma, label_pred, label_emp):
        x_max = max(float(np.max(np.abs(scores))), abs(mean_pred) + 4.0 * sigma) * 1.05
        x_lo, x_hi = mean_pred - x_max, mean_pred + x_max
        x_lo = min(x_lo, scores.min() - 0.1 * (scores.max() - scores.min() + 1e-9))
        x_hi = max(x_hi, scores.max() + 0.1 * (scores.max() - scores.min() + 1e-9))
        bins = np.linspace(x_lo, x_hi, 70)
        ax.hist(
            scores, bins=bins, density=True,
            color='#7DBEDC', edgecolor='#1F4E79', linewidth=0.6, alpha=0.75,
            label=label_emp,
        )
        zz = np.linspace(x_lo, x_hi, 400)
        pdf = np.exp(-0.5 * ((zz - mean_pred) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        ax.plot(zz, pdf, color='#cc4422', linewidth=2.4, linestyle='--', label=label_pred)
        ax.axvline(mean_pred, color='#cc4422', linewidth=1.2, linestyle=':', alpha=0.7)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel('Score (NDIS scale)')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right', fontsize=14)

    base = os.path.join(fig_dir, f'gaussianity_{mech_label.lower()}_eps{eps_tag}')
    with plt.rc_context(_RC):
        fig_out, ax_out = plt.subplots(figsize=(11, 6.5))
        _hist_with_pdf(
            ax_out, out_scores, mean_pred=0.0, sigma=sigma_pred,
            label_pred=fr'$\mathcal{{N}}(0, \sigma^2)$,  $\sigma={sigma_pred:.3f}$',
            label_emp=fr'OUT  ($\hat\mu={out_mu:.3f},\, \hat\sigma={out_sd:.3f}$)',
        )
        fig_out.tight_layout()
        fig_out.savefig(f'{base}_out.png', dpi=300, bbox_inches='tight')
        fig_out.savefig(f'{base}_out.pdf',                bbox_inches='tight')

        fig_in, ax_in = plt.subplots(figsize=(11, 6.5))
        _hist_with_pdf(
            ax_in, in_scores, mean_pred=mean_in_pred, sigma=sigma_pred,
            label_pred=(fr'$\mathcal{{N}}(\mu, \sigma^2)$,'
                        fr' $\mu={mean_in_pred:.3f}$ ({in_label}),'
                        fr' $\sigma={sigma_pred:.3f}$'),
            label_emp=fr'IN  ($\hat\mu={in_mu:.3f},\, \hat\sigma={in_sd:.3f}$)',
        )
        fig_in.tight_layout()
        fig_in.savefig(f'{base}_in.png', dpi=300, bbox_inches='tight')
        fig_in.savefig(f'{base}_in.pdf',                bbox_inches='tight')

    print(f"Saved Gaussianity diagnostics to: {base}_{{in,out}}.{{png,pdf}}")


def main():
    parser = argparse.ArgumentParser(description='Run auditing methods comparison')
    parser.add_argument('--exp-dir', type=str, help='Single experiment directory')
    parser.add_argument('--exp-dirs', type=str, nargs='+', help='Multiple experiment directories')
    parser.add_argument('--complexity', action='store_true', help='Run sample complexity analysis')
    parser.add_argument('--ablation-T', action='store_true',
                        help='With --exp-dirs: sweep training-step count T at fixed target eps.')
    parser.add_argument('--with-andrew', action='store_true',
                        help='Include Andrew et al. 2024 (max-over-iterates cosine). '
                             'DP-FTRL only — DP-SGD runs do not save in_scores_andrew_*.csv.')
    parser.add_argument('--independence', action='store_true',
                        help='With --exp-dir: render the canary-direction Gram matrix + '
                             'per-canary score density (independence diagnostic).')
    parser.add_argument('--gaussianity', action='store_true',
                        help='With --exp-dir: side-by-side IN / OUT score histograms '
                             'against the asymptotic Gaussians predicted by the noise '
                             'calibration. IN mean is sqrt(L)*C (DP-FTRL) or q*sqrt(T)*C '
                             '(DP-SGD).')
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--significance', type=float, default=0.05)
    parser.add_argument('--fig-dir', type=str, default=None)
    args = parser.parse_args()

    if args.fig_dir is None:
        args.fig_dir = os.path.join(project_dir, 'fig')
    os.makedirs(args.fig_dir, exist_ok=True)

    if args.independence and args.exp_dir:
        run_independence(args.exp_dir, args.fig_dir, with_andrew=args.with_andrew)
    elif args.gaussianity and args.exp_dir:
        run_gaussianity(args.exp_dir, args.fig_dir)
    elif args.ablation_T and args.exp_dirs:
        run_ablation_T(args.exp_dirs, args.delta, args.significance, args.fig_dir,
                       with_andrew=args.with_andrew)
    elif args.complexity and args.exp_dir:
        run_complexity(args.exp_dir, args.delta, args.significance, args.fig_dir)
    elif args.exp_dirs:
        run_multi(args.exp_dirs, args.delta, args.significance, args.fig_dir,
                  with_andrew=args.with_andrew)
    elif args.exp_dir:
        run_single(args.exp_dir, args.delta, args.significance, args.fig_dir,
                   with_andrew=args.with_andrew)
    else:
        parser.error(
            "Provide --exp-dir or --exp-dirs. Add --complexity for sample size analysis, "
            "--ablation-T to sweep T at fixed eps, --independence for the canary "
            "orthogonality heatmap, or --gaussianity for the score-Gaussian fit."
        )


if __name__ == '__main__':
    main()