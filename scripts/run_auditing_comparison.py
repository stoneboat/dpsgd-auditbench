#!/usr/bin/env python3
"""
Run all 3 auditing methods on saved whitebox scores and plot comparison.

Methods:
  1. Steinke et al. 2023 (one-run, Theorem 5.2) — uses raw sum scores
  2. Mahloujifar et al. 2025 (f-DP) — uses raw sum scores
  3. NDIS (normal distribution indistinguishability spectrum) — uses normalized scores

Usage:
  # Single experiment directory:
  python scripts/run_auditing_comparison.py --exp-dir ./data/mislabeled-canaries-<seed>-5000-0.5-cifar10

  # Multiple experiment directories (one per epsilon), final-epoch bar chart & ablation:
  python scripts/run_auditing_comparison.py --exp-dirs ./data/exp_eps1 ./data/exp_eps2 ./data/exp_eps4 ./data/exp_eps8

  # Sample-complexity sweep on a single experiment (Generates Main + Ablation plots):
  python scripts/run_auditing_comparison.py --complexity --exp-dir ./data/mislabeled-canaries-<seed>-5000-0.5-cifar10
"""

import argparse
import csv
import json
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)

from auditing import CanaryScoreAuditor
from canary_score_diagnostics import dirac_canary_interference
from whitebox_auditing.ndis_1d import (
    _ndis_eps_from_moments,
    ndis_eps_lb_all,
    ndis_eps_lb_parametric_bonferroni,
)
from whitebox_auditing.tree_mechanism import num_levels

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
    'fdp':     {'color': '#2ca02c', 'marker': 's', 'linestyle': '--',  'linewidth': 2.4, 'markersize': 7,  'zorder': 3, 'label': 'Mahloujifar et al. 2025 (f-DP)'},
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
CANARY_BUDGETS = (50, 100, 200, 400, 600, 1000, 1500, 2000, 3500, 5000)
COMPLEXITY_TRIALS = 50


def _plot_method(ax, x, y, key, label_override=None):
    """Plots a method series, allowing for dynamic legend names via label_override."""
    s = _STYLE[key]
    label = label_override if label_override is not None else s['label']
    ax.plot(x, y,
            color=s['color'], marker=s['marker'], linestyle=s['linestyle'],
            linewidth=s['linewidth'], markersize=s['markersize'],
            label=label, zorder=s['zorder'],
            markerfacecolor=s['color'], markeredgecolor='white', markeredgewidth=0.6)


def print_run_config(exp_dir):
    """One-line identification of the run being audited.

    Experiment directories are named by database seed, so the path alone does
    not say which hyperparameter cell a run belongs to. Print the fields that
    define the cell (batch size, clipping norm, noise multiplier, steps) so a
    sequence of --exp-dir invocations is self-labelling.
    """
    hparams_path = os.path.join(exp_dir, 'hparams.json')
    if not os.path.isfile(hparams_path):
        print(f"Run config: no hparams.json in {exp_dir}")
        return
    with open(hparams_path) as f:
        hp = json.load(f)

    fields = [
        ('eps', 'epsilon', '{:g}'),
        ('B', 'logical_batch_size', '{:d}'),
        ('C', 'max_grad_norm', '{:g}'),
        ('sigma', 'noise_multiplier', '{:.4f}'),
        ('T', 'target_steps', '{:d}'),
        ('K', 'aug_multiplicity', '{:d}'),
        ('m', 'canary_count', '{:d}'),
    ]
    parts = []
    for label, key, fmt in fields:
        if hp.get(key) is not None:
            parts.append(f"{label}={fmt.format(hp[key])}")
    # Only present when the run was produced with --fault (see src/faults.py).
    if hp.get('fault') not in (None, 'none'):
        parts.append(f"fault={hp['fault']}")
        if hp.get('eps_true') is not None:
            parts.append(f"eps_true={hp['eps_true']:.2f}")

    print(f"Run config: {', '.join(parts)}")
    print(f"  dir: {exp_dir}")


def _resolve_score_path(exp_dir, kind, side, epoch):
    candidates = [
        os.path.join(exp_dir, f'{side}_scores_{kind}_{epoch:06d}.csv'),
        os.path.join(exp_dir, f'{side}_scores_optimal_{epoch:06d}.csv'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _audit_thresholds(in_sum, out_sum, delta, significance):
    auditor = CanaryScoreAuditor(in_sum, out_sum)
    eps_steinke, _ = auditor._epsilon_one_run_all_thresholds(
        significance=significance, delta=delta, one_sided=True, threshold=None, use_fdp=False,
    )
    eps_fdp, _ = auditor._epsilon_one_run_all_thresholds(
        significance=significance, delta=delta, one_sided=True, threshold=None, use_fdp=True,
    )
    return {'steinke': eps_steinke, 'fdp': eps_fdp}


def audit_epoch(exp_dir, epoch, delta, significance, method=None, target_eps=None,
                with_andrew=False, *, steps=None, noise_multiplier_floor=None):
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
    if noise_multiplier_floor is not None:
        _, steps, in_ndis, out_ndis = _load_finite_t_scores(exp_dir, steps, epoch=epoch)
    else:
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

    out = {'upper': eps_upper, **_audit_thresholds(in_sum, out_sum, delta, significance)}
    if noise_multiplier_floor is not None:
        correction_r = _gaussian_correction_for_noise_floor(steps, noise_multiplier_floor)
        out['correction'] = _corrected_bootstrap(
            in_ndis, out_ndis, steps, delta, significance, include_bonferroni=True, correction_r=correction_r,
        )
    else:
        out.update(dict.fromkeys(NDIS_KEYS, 0.0))
        try:
            all_out = ndis_eps_lb_all(
                in_ndis, out_ndis, delta=delta,
                alpha=significance, n_bootstrap=2000,
                pool_variance=False,
            )
            for m in NDIS_METHODS:
                if m in all_out:
                    out[f'ndis_{m}'] = float(all_out[m]['eps_lb'])
        except (ValueError, RuntimeError) as e:
            print(f"  NDIS failed ({e}), setting all variants to 0")

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
    print_run_config(exp_dir)
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

        print_run_config(exp_dir)
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
            geometry = dirac_canary_interference(indices)
            ortho_results.append({
                'target': target_eps,
                'm': geometry['m'],
                'collisions': geometry['collision_count'],
                'orthogonal': geometry['collision_count'] == 0,
                'mutual_coherence': geometry['mutual_coherence'],
                'rms_offdiag': geometry['rms_offdiag'],
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
        print(
            f"{'Target Eps':>10} | {'Canaries (m)':>12} | {'Collisions':>10} | "
            f"{'Coherence':>10} | {'RMS offdiag':>11} | Status"
        )
        print("-" * 60)
        for res in ortho_results:
            if res:
                status = "PASS (IID)" if res['orthogonal'] else "FAIL (CORRELATED)"
                print(
                    f"{res['target']:10.2f} | {res['m']:12d} | "
                    f"{res['collisions']:10d} | "
                    f"{res['mutual_coherence']:10.3e} | "
                    f"{res['rms_offdiag']:11.3e} | {status}"
                )
        print("="*60)
        print("Note: Dirac canaries are perfectly orthogonal if collisions = 0.")

    _plot_multi(series, fig_dir, with_andrew)


def _plot_multi(series, fig_dir, with_andrew=False):
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

    _plot_training_steps(series, target_eps_label, fig_dir, with_andrew)


def _plot_training_steps(series, target_eps_label, fig_dir, with_andrew=False):
    eps_tag = f"{target_eps_label:g}".replace('.', 'p')
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

    total_budgets = CANARY_BUDGETS
    full_budget = len(in_sum) + len(out_sum)

    metric_keys = ('steinke', 'fdp', *NDIS_KEYS)
    results = {k: [] for k in metric_keys}
    n_trials = COMPLEXITY_TRIALS

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

            baselines = _audit_thresholds(s_in, s_out, delta, significance)
            try:
                ndis_out = ndis_eps_lb_all(
                    n_in_scores, n_out_scores, delta=delta,
                    alpha=significance, pool_variance=True,
                )
            except (ValueError, RuntimeError):
                ndis_out = None

            trial_data['steinke'].append(baselines['steinke'])
            trial_data['fdp'].append(baselines['fdp'])
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

    _plot_sample_complexity(total_budgets, results, target_eps, fig_dir)


def _plot_sample_complexity(total_budgets, results, target_eps, fig_dir):
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

    geometry = dirac_canary_interference(coords)
    print(
        f"Dirac geometry (m={m_total}): "
        f"collisions={geometry['collision_count']}, "
        f"mutual_coherence={geometry['mutual_coherence']:.3e}, "
        f"RMS_offdiag={geometry['rms_offdiag']:.3e}"
    )
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
        ax.set_xlabel('Normalized Scores')
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


def _load_finite_t_scores(exp_dir, steps, *, epoch=None):
    if steps is not None and steps < 1:
        raise ValueError('--steps must be positive')

    if epoch is None:
        epochs = [int(match[1]) for name in os.listdir(exp_dir) if (match := re.fullmatch(r'in_scores_ndis_(\d+)\.csv', name))]
        if not epochs:
            raise ValueError('no saved IN NDIS scores found')
        epoch = max(epochs)
    if os.path.isfile(os.path.join(exp_dir, f'in_scores_andrew_{epoch:06d}.csv')):
        raise ValueError('finite-T NDIS requires DP-SGD Dirac scores')
    in_scores, out_scores = [
        np.loadtxt(os.path.join(exp_dir, f'{side}_scores_ndis_{epoch:06d}.csv'), delimiter=',', ndmin=1)
        for side in ('in', 'out')
    ]
    if any(scores.ndim != 1 or scores.size < 2 or not np.isfinite(scores).all() for scores in (in_scores, out_scores)):
        raise ValueError('each NDIS score file must contain at least two finite scores')

    #Recover T from the sum and the scores
    sum_path = os.path.join(exp_dir, f'in_scores_sum_{epoch:06d}.csv')
    if os.path.isfile(sum_path):
        in_sum = np.loadtxt(sum_path, delimiter=',', ndmin=1)
        if in_sum.shape != in_scores.shape or not np.isfinite(in_sum).all():
            raise ValueError('saved IN sum and NDIS scores must have matching shapes and finite values')
        if steps is None:
            largest = np.argmax(np.abs(in_scores))
            if in_scores[largest] == 0:
                raise ValueError('pass --steps when the IN scores are all zero')
            steps = round((in_sum[largest] / in_scores[largest])**2)
        if steps < 1 or not np.allclose(in_sum, in_scores * math.sqrt(steps), rtol=1e-6, atol=1e-10):
            raise ValueError('the saved IN sum and NDIS scores disagree with the checkpoint step count')
    elif steps is None:
        raise ValueError('pass --steps when the saved IN sum scores are unavailable')

    return epoch, steps, in_scores, out_scores


def run_gaussian_correction(exp_dir, delta, significance, steps, correction_r, *, noise_multiplier_floor=None):
    if not 0 < delta < 1 or not 0 < significance < 1:
        raise ValueError('Gaussian correction requires delta and significance in (0, 1)')
    epoch, steps, in_scores, out_scores = _load_finite_t_scores(exp_dir, steps)
    if noise_multiplier_floor is not None:
        correction_r = _gaussian_correction_for_noise_floor(steps, noise_multiplier_floor)
    estimates = _corrected_bootstrap(in_scores, out_scores, steps, delta, significance,
                                     include_bonferroni=True, correction_r=correction_r)
    print(f'Checkpoint {epoch}, T={steps}, delta={delta:g}, confidence level={1 - significance:.1%}')
    print(f'IN probability correction r={correction_r:.6g}')
    print('\n| NDIS estimate | Gaussian epsilon | Corrected epsilon |')
    print('| --- | --- | --- |')
    print(f'| Bootstrap IN -> OUT | {estimates["bootstrap_eps"]:.6f} | {estimates["corrected_bootstrap_eps"]:.6f} |')
    print(f'| Bonferroni IN -> OUT | {estimates["ndis_parametric_bonferroni"]:.6f} | {estimates["ndis_parametric_bonferroni_corrected"]:.6f} |')


def _gaussian_correction_for_noise_floor(steps, noise_multiplier_floor):
    if not math.isfinite(noise_multiplier_floor) or noise_multiplier_floor <= 0:
        raise ValueError('noise multiplier floor must be finite and positive')
    # Uniform over q in [0, 1], with two endpoints in the Gaussian decision region.
    skew_term = 1 / (18 * math.sqrt(6 * math.pi * steps) * noise_multiplier_floor**3)
    remainder = 1 / (24 * math.pi * steps * noise_multiplier_floor**4)
    return math.ceil((skew_term + remainder) * 1e8) / 1e8


def _corrected_bootstrap(in_scores, out_scores, steps, delta, significance, *,
                         include_bonferroni=False, correction_r):
    estimates = {
        'T': steps, 'delta': delta, 'alpha': significance, 'confidence': 1 - significance,
        'n_in': in_scores.size, 'n_out': out_scores.size, 'm': in_scores.size + out_scores.size,
        'gaussian_correction_r': correction_r,
    }
    ndis = ndis_eps_lb_all(
        in_scores, out_scores, delta=delta, alpha=significance,
        rng=np.random.default_rng(2), methods=NDIS_METHODS if include_bonferroni else ('bootstrap_ellipsoid',),
        gaussian_correction_r=correction_r,
    )
    bootstrap = ndis['bootstrap_ellipsoid']
    if not math.isfinite(bootstrap['eps_lb']):
        raise ValueError('bootstrap NDIS returned a nonfinite estimate')
    if include_bonferroni:
        bonferroni = ndis['parametric_bonferroni']
        estimates['ndis_parametric_bonferroni'] = bonferroni['eps_lb']
        estimates['ndis_parametric_bonferroni_corrected'] = bonferroni['eps_lb_corrected']
    estimates['bootstrap_pool_variance'] = False
    estimates['bootstrap_eps'] = bootstrap['eps_lb']
    estimates['corrected_bootstrap_eps'] = bootstrap['eps_lb_corrected']
    return estimates


def _finite_t_run_metadata(exp_dir):
    path = os.path.join(exp_dir, 'hparams.json')
    metadata = {}
    if os.path.isfile(path):
        with open(path) as source:
            metadata = json.load(source)
        if not isinstance(metadata, dict):
            raise ValueError('hparams.json must contain a JSON object')
    fields = ('target_steps', 'logical_batch_size', 'max_grad_norm', 'lr', 'adaptive_clipping', 'fault', 'eps_claimed', 'eps_true')
    return {'exp_dir': exp_dir, 'target_epsilon': metadata.get('epsilon'), 'ndis_direction': 'in_to_out',
            **{key: metadata.get(key) for key in fields}}


def _write_corrected_bootstrap_csv(path, rows):
    saved = []
    with open(path, 'w', newline='') as output:
        writer = None
        for row in rows:
            if writer is None:
                writer = csv.DictWriter(output, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            output.flush()
            saved.append(row)
    return saved


def _plot_corrected_bootstrap(rows, axis, fig_dir):
    x_field = 'target_steps' if axis == 'T' else axis
    if any(row[x_field] is None for row in rows):
        raise ValueError(f'The original plot requires {x_field} for every run')
    rows = sorted(rows, key=lambda row: row[x_field])
    x = [row[x_field] for row in rows]
    series = {key: [row[key] for row in rows] for key in ('steinke', 'fdp')}
    series['ndis_parametric_bonferroni'] = [row['ndis_parametric_bonferroni_corrected'] for row in rows]
    series['ndis_bootstrap_ellipsoid'] = [row['corrected_bootstrap_eps'] for row in rows]
    target_eps = rows[0]['target_epsilon']
    if axis == 'm':
        _plot_sample_complexity(x, series, target_eps, fig_dir)
        return

    series['upper'] = [row['upper'] for row in rows]
    if axis == 'T':
        if target_eps is None:
            raise ValueError('The training-step plot requires epsilon in hparams.json')
        series['T'] = x
        _plot_training_steps(series, target_eps, fig_dir)
    else:
        series['target'] = x
        _plot_multi(series, fig_dir)


def run_corrected_bootstrap_report(exp_dirs, delta, significance, fig_dir, *, noise_floors, steps=None, axis=None):
    failures = []

    def evaluate_runs():
        for exp_dir in dict.fromkeys(map(os.path.realpath, exp_dirs)):
            try:
                metadata = _finite_t_run_metadata(exp_dir)
                target_eps = metadata['target_epsilon']
                noise_floor = noise_floors.get(exp_dir, noise_floors.get(target_eps))
                if noise_floor is None:
                    raise ValueError('no noise multiplier floor specified for this run')
                epoch = get_final_epoch(exp_dir)
                if epoch is None:
                    raise ValueError('The comparison requires saved canary sums')
                if metadata['adaptive_clipping'] or exp_dir.endswith('-adaptive'):
                    clip_norms = os.path.join(exp_dir, f'clip_norms_{epoch:06d}.csv')
                    if not os.path.isfile(clip_norms):
                        raise ValueError('adaptive clipping requires scores normalized by the per-step clipping bounds')
                metrics = audit_epoch(exp_dir, epoch, delta, significance, steps=steps,
                                      noise_multiplier_floor=noise_floor)
                if metrics is None:
                    raise ValueError(f'Missing IN or OUT scores for checkpoint {epoch}')
                estimates = metrics.pop('correction')
                estimates.update(metrics)
                estimates['noise_multiplier_floor'] = noise_floor
                print(f'Target epsilon {target_eps}, checkpoint {epoch}, T={estimates["T"]}, '
                      f'noise multiplier floor={noise_floor:g}, r={estimates["gaussian_correction_r"]:.6g}, '
                      f'corrected bootstrap epsilon={estimates["corrected_bootstrap_eps"]:.6f}')
            except (OSError, ValueError, RuntimeError) as error:
                print(f'{exp_dir}: {error}', file=sys.stderr)
                failures.append(exp_dir)
                continue
            yield {**metadata, 'checkpoint': epoch, **estimates}

    path = os.path.join(fig_dir, 'corrected_bootstrap.csv')
    rows = _write_corrected_bootstrap_csv(path, evaluate_runs())
    print(f'Saved {len(rows)} rows to {path}')
    if rows and axis:
        _plot_corrected_bootstrap(rows, axis, fig_dir)
    return bool(rows) and not failures


def _complexity_trial_parts(directory):
    return sorted(os.path.join(directory, name) for name in os.listdir(directory)
                  if re.fullmatch(r'sample_complexity_trials_part_\d+\.csv', name))


def _read_complexity_trials(paths, trial_indices, settings, metric_names):
    trials = {}
    for path in paths:
        with open(path, newline='') as source:
            # Another job may still be flushing the final row.
            complete_rows = source.read().rpartition('\n')[0]
        for row in csv.DictReader(complete_rows.splitlines()):
            if row.get('ndis_direction') != settings['ndis_direction']:
                raise ValueError(f'{path}: cached NDIS estimates must use IN-to-OUT only. Start a fresh canary sweep.')
            key = int(row['m']), int(row['trial'])
            if key in trials or key not in trial_indices:
                raise ValueError(f'{path}: unexpected or duplicate trial {key}')
            if any(row.get(name) != ('' if value is None else str(value)) for name, value in settings.items()):
                raise ValueError(f'{path}: trial {key} has different audit settings')
            in_indices, out_indices = trial_indices[key]
            if (int(row['n_in']), int(row['n_out'])) != (len(in_indices), len(out_indices)):
                raise ValueError(f'{path}: trial {key} has a different IN/OUT split')
            row.update(settings)
            row.update(m=key[0], trial=key[1], n_in=len(in_indices), n_out=len(out_indices))
            for name in metric_names:
                if name == 'ndis_parametric_bonferroni_corrected' and not row.get(name):
                    row.pop(name, None)
                else:
                    row[name] = float(row[name])
            trials[key] = row
    return trials


def run_corrected_bootstrap_complexity(exp_dir, delta, significance, fig_dir, n_trials, *, noise_multiplier_floor,
                                       steps=None, csv_only=False, shard=None,
                                       resume_csv=None, merge_shards=False):
    if n_trials < 1:
        raise ValueError('--complexity-trials must be positive')
    if shard is not None and (merge_shards or not 0 <= shard[0] < shard[1]):
        raise ValueError('invalid canary sweep shard')
    exp_dir = os.path.realpath(exp_dir)
    epoch, steps, in_scores, out_scores = _load_finite_t_scores(exp_dir, steps)
    metadata = _finite_t_run_metadata(exp_dir)
    correction_r = _gaussian_correction_for_noise_floor(steps, noise_multiplier_floor)
    metadata['noise_multiplier_floor'] = noise_multiplier_floor
    metadata['gaussian_correction_r'] = correction_r
    if not csv_only:
        in_sum, out_sum = [
            np.loadtxt(os.path.join(exp_dir, f'{side}_scores_sum_{epoch:06d}.csv'), delimiter=',', ndmin=1)
            for side in ('in', 'out')
        ]
        if in_sum.shape != in_scores.shape or out_sum.shape != out_scores.shape:
            raise ValueError('Saved NDIS scores and sums must have matching shapes')
    full_budget = in_scores.size + out_scores.size
    budgets = [budget for budget in CANARY_BUDGETS if budget < full_budget] + [full_budget]
    rng = np.random.default_rng(2)

    def evaluate_trial(in_indices, out_indices):
        estimates = _corrected_bootstrap(
            in_scores[in_indices], out_scores[out_indices], steps, delta, significance,
            include_bonferroni=not csv_only, correction_r=correction_r,
        )
        if not csv_only:
            estimates.update(_audit_thresholds(in_sum[in_indices], out_sum[out_indices], delta, significance))
        return estimates

    trial_indices = {}
    for budget in budgets[:-1]:
        min_in = max(2, budget - out_scores.size)
        max_in = min(in_scores.size, budget - 2)
        for trial in range(1, n_trials + 1):
            n_in = int(np.clip(rng.binomial(budget, 0.5), min_in, max_in))
            in_indices = rng.choice(in_scores.size, n_in, replace=False)
            out_indices = rng.choice(out_scores.size, budget - n_in, replace=False)
            trial_indices[budget, trial] = in_indices, out_indices
    trial_indices[full_budget, 1] = np.arange(in_scores.size), np.arange(out_scores.size)

    mean_fields = ('bootstrap_eps', 'corrected_bootstrap_eps')
    if not csv_only:
        mean_fields += ('steinke', 'fdp', 'ndis_parametric_bonferroni', 'ndis_parametric_bonferroni_corrected')

    settings = {**metadata, 'checkpoint': epoch, 'T': steps, 'delta': delta, 'alpha': significance,
                'confidence': 1 - significance, 'bootstrap_pool_variance': False}
    trial_path = os.path.join(fig_dir, 'sample_complexity_trials.csv')
    if merge_shards:
        saved = _read_complexity_trials(_complexity_trial_parts(fig_dir), trial_indices, settings, mean_fields)
        if len(saved) != len(trial_indices):
            raise ValueError(f'canary sweep incomplete: found {len(saved)} of {len(trial_indices)} trials')
        if not csv_only and any('ndis_parametric_bonferroni_corrected' not in row for row in saved.values()):
            raise ValueError('Some shards lack corrected Bonferroni values. Resume from this directory to fill them in.')
        trials = _write_corrected_bootstrap_csv(trial_path, (saved[key] for key in trial_indices))
    else:
        selected = list(trial_indices)
        if shard is not None:
            index, count = shard
            selected = selected[index::count]
            trial_path = os.path.join(fig_dir, f'sample_complexity_trials_part_{index:03d}.csv')
        saved = {}
        if resume_csv:
            if os.path.isdir(resume_csv):
                paths = _complexity_trial_parts(resume_csv) or [os.path.join(resume_csv, 'sample_complexity_trials.csv')]
                saved = _read_complexity_trials(paths, trial_indices, settings, mean_fields)
            else:
                saved = _read_complexity_trials([resume_csv], trial_indices, settings, mean_fields)
                if list(saved) != list(trial_indices)[:len(saved)]:
                    raise ValueError('resume CSV must contain the initial trials of the same canary sweep')
        if shard is not None and os.path.isfile(trial_path):
            previous = _read_complexity_trials([trial_path], trial_indices, settings, mean_fields)
            saved.update(previous)
        reused = sum(key in saved for key in selected)
        bonferroni_to_fill = 0 if csv_only else sum(
            'ndis_parametric_bonferroni_corrected' not in saved[key] for key in selected if key in saved
        )
        print(f'Canary trials assigned={len(selected)}, reusing={reused}, remaining={len(selected) - reused}, '
              f'Bonferroni to fill={bonferroni_to_fill}')

        def evaluate_subsets():
            for key in selected:
                if key in saved:
                    row = saved[key]
                    if not csv_only and 'ndis_parametric_bonferroni_corrected' not in row:
                        in_indices, out_indices = trial_indices[key]
                        bonferroni = ndis_eps_lb_parametric_bonferroni(
                            in_scores[in_indices], out_scores[out_indices], delta, alpha=significance,
                            gaussian_correction_r=correction_r,
                        )
                        row['ndis_parametric_bonferroni_corrected'] = bonferroni['eps_lb']
                    yield row
                else:
                    estimates = evaluate_trial(*trial_indices[key])
                    yield {**metadata, 'checkpoint': epoch, 'trial': key[1], **estimates}

        trials = _write_corrected_bootstrap_csv(trial_path, evaluate_subsets())
        if shard is not None:
            print(f'Saved {len(trials)} trials to {trial_path}')
            return

    summary = []
    for budget in budgets:
        rows = [row for row in trials if row['m'] == budget]
        means = {key: np.mean([row[key] for row in rows]) for key in mean_fields}
        summary.append({**metadata, 'checkpoint': epoch, 'T': steps, 'm': budget, 'trials': len(rows),
                        'delta': delta, 'alpha': significance, 'confidence': 1 - significance,
                        'bootstrap_pool_variance': rows[0]['bootstrap_pool_variance'], **means})
    path = os.path.join(fig_dir, 'sample_complexity.csv')
    _write_corrected_bootstrap_csv(path, summary)
    if not csv_only:
        _plot_corrected_bootstrap(summary, 'm', fig_dir)
    print(f'Saved {len(trials)} trials to {trial_path} and {len(summary)} rows to {path}')


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
    parser.add_argument('--compare-tails', action='store_true',
                        help='Compare Gaussian and finite-T corrected NDIS on DP-SGD Dirac scores.')
    correction = parser.add_mutually_exclusive_group()
    correction.add_argument('--gaussian-correction-r', type=float,
                            help='Supplied bound on the IN decision-region probability error for a single --compare-tails run.')
    correction.add_argument('--noise-multiplier-floor', type=float,
                            help='Supplied lower bound on the noise multiplier, used to compute r for --compare-tails.')
    parser.add_argument('--steps', type=int, help='Checkpoint step count for --compare-tails. Defaults to the count inferred from the saved IN sum.')
    parser.add_argument('--csv-only', action='store_true',
                        help='Skip corrected NDIS plots. For sample complexity, export bootstrap estimates only.')
    parser.add_argument('--complexity-trials', type=int, default=COMPLEXITY_TRIALS,
                        help='Subsamples per budget for --compare-tails --complexity (default: %(default)s).')
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

    has_correction = args.gaussian_correction_r is not None or args.noise_multiplier_floor is not None
    if has_correction and not args.compare_tails:
        parser.error('Gaussian correction inputs require --compare-tails')

    if args.compare_tails:
        if not has_correction:
            parser.error('--compare-tails requires --gaussian-correction-r or --noise-multiplier-floor')
        if bool(args.exp_dir) == bool(args.exp_dirs):
            parser.error('--compare-tails requires either --exp-dir or --exp-dirs')
        if args.exp_dirs and args.steps is not None:
            parser.error('--compare-tails --steps requires a single --exp-dir')
        if args.complexity and (args.exp_dirs or args.ablation_T):
            parser.error('--compare-tails --complexity requires a single --exp-dir')
        try:
            export = args.exp_dirs or args.fig_dir or args.complexity or args.ablation_T or args.csv_only
            if export and args.gaussian_correction_r is not None:
                parser.error('--compare-tails exports require --noise-multiplier-floor')
            if not export:
                run_gaussian_correction(args.exp_dir, args.delta, args.significance, args.steps,
                                        args.gaussian_correction_r, noise_multiplier_floor=args.noise_multiplier_floor)
                return

            fig_dir = args.fig_dir or os.path.join(project_dir, 'fig', 'fixed_gaussianity_error')
            os.makedirs(fig_dir, exist_ok=True)
            if args.complexity:
                run_corrected_bootstrap_complexity(args.exp_dir, args.delta, args.significance, fig_dir,
                                                   args.complexity_trials, steps=args.steps, csv_only=args.csv_only,
                                                   noise_multiplier_floor=args.noise_multiplier_floor)
            else:
                axis = None if args.csv_only else ('T' if args.ablation_T else 'target_epsilon')
                exp_dirs = args.exp_dirs or [args.exp_dir]
                noise_floors = {os.path.realpath(path): args.noise_multiplier_floor for path in exp_dirs}
                if not run_corrected_bootstrap_report(exp_dirs, args.delta, args.significance, fig_dir,
                                                      steps=args.steps, axis=axis, noise_floors=noise_floors):
                    sys.exit(1)
        except (OSError, ValueError, RuntimeError) as e:
            parser.error(str(e))
        return

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
