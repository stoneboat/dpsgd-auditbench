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
"""

import sys
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)

from auditing import CanaryScoreAuditor, MultiSplit
from whitebox_auditing.ndis_1d import (
    ndis_eps_from_delta_1d_brentq,
    ndis_eps_lower_bound_with_ci,
    estimate_mean_variance,
)


def audit_epoch(exp_dir, epoch, delta, significance):
    """Run all 3 auditing methods for a given epoch. Returns dict of results."""
    in_sum_path = os.path.join(exp_dir, f'in_scores_sum_{epoch:06d}.csv')
    out_sum_path = os.path.join(exp_dir, f'out_scores_sum_{epoch:06d}.csv')
    in_ndis_path = os.path.join(exp_dir, f'in_scores_ndis_{epoch:06d}.csv')
    out_ndis_path = os.path.join(exp_dir, f'out_scores_ndis_{epoch:06d}.csv')
    privacy_path = os.path.join(exp_dir, f'privacy_params_{epoch:06d}.csv')

    if not all(os.path.isfile(p) for p in [in_sum_path, out_sum_path, in_ndis_path, out_ndis_path]):
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
    # Default threshold_strategy in auditing.py is Bonferroni, which divides
    # significance by len(thresholds) ~ m = 5000 -> alpha effectively 1e-5
    # for m=5000. Mahloujifar's reported numbers do not apply this correction
    # (they post-hoc pick the best c'), so for parity we use a held-out split:
    # half the canaries pick the threshold, half compute eps. No multiplicity
    # correction needed because only one threshold is evaluated on the held-out
    # half. MultiSplit reduces variance by averaging multiple random splits.
    threshold_strategy = MultiSplit(num_samples=10, threshold_estimation_frac=0.5, seed=0)
    auditor = CanaryScoreAuditor(in_sum, out_sum)
    eps_steinke = auditor.epsilon_one_run(
        significance=significance, delta=delta,
        threshold_strategy=threshold_strategy,
    )
    eps_fdp = auditor.epsilon_one_run_fdp(
        significance=significance, delta=delta,
        threshold_strategy=threshold_strategy,
    )

    # NDIS: use the bootstrap CI lower bound with pool_variance=True (the
    # equal-variance case for the gradient-projection score). The earlier
    # ndis_eps_from_delta_1d_brentq call on raw sample moments was a biased
    # POINT estimate; ndis_eps_lower_bound_with_ci is the audit-valid LB.
    try:
        eps_ndis = ndis_eps_lower_bound_with_ci(
            in_ndis, out_ndis, delta=delta,
            alpha=significance, n_bootstrap=2000,
            pool_variance=True,
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
    """Find the largest epoch with score files."""
    epochs = sorted(set(
        int(f.split('_')[-1].replace('.csv', ''))
        for f in os.listdir(exp_dir)
        if f.startswith('in_scores_sum_') and f.endswith('.csv')
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
        if f.startswith('in_scores_sum_') and f.endswith('.csv')
    ))
    if not epochs:
        print(f"Error: no in_scores_sum_*.csv files found in {exp_dir}")
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
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epoch_list, upper_bounds, 'b-o', label='Upper bound (RDP)', linewidth=2)
    ax.plot(epoch_list, steinke_bounds, 'r-s', label='Steinke 2023 (one-run)', linewidth=2)
    ax.plot(epoch_list, fdp_bounds, 'g-^', label='Mahloujifar 2024 (f-DP)', linewidth=2)
    ax.plot(epoch_list, ndis_bounds, 'm-D', label='NDIS (ours)', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Epsilon', fontsize=12)
    ax.set_title('Privacy Bounds Comparison: Upper (theoretical) vs Lower (empirical)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(fig_dir, 'privacy_bounds_comparison.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")


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

    # Line plot (same style as single-experiment plot, but x-axis = target epsilon)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(target_epsilons, upper_bounds, 'b-o', label='Upper bound (RDP)', linewidth=2)
    ax.plot(target_epsilons, steinke_bounds, 'r-s', label='Steinke 2023 (one-run)', linewidth=2)
    ax.plot(target_epsilons, fdp_bounds, 'g-^', label='Mahloujifar 2024 (f-DP)', linewidth=2)
    ax.plot(target_epsilons, ndis_bounds, 'm-D', label='NDIS (ours)', linewidth=2)
    ax.set_xlabel('Target Epsilon', fontsize=12)
    ax.set_ylabel('Epsilon', fontsize=12)
    ax.set_title('Privacy Bounds Comparison: Upper (theoretical) vs Lower (empirical)', fontsize=14)
    ax.set_xticks(target_epsilons)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_path = os.path.join(fig_dir, 'privacy_bounds_comparison_multi_eps.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description='Run 3 auditing methods on whitebox scores')
    parser.add_argument('--exp-dir', type=str, default=None,
                        help='Single experiment directory (per-epoch line plot)')
    parser.add_argument('--exp-dirs', type=str, nargs='+', default=None,
                        help='Multiple experiment directories (final-epoch bar chart)')
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--significance', type=float, default=0.05,
                        help='Significance level for paper methods (95%% confidence)')
    parser.add_argument('--fig-dir', type=str, default=None,
                        help='Directory to save figures (default: <project>/fig)')
    args = parser.parse_args()

    if args.fig_dir is None:
        args.fig_dir = os.path.join(project_dir, 'fig')
    os.makedirs(args.fig_dir, exist_ok=True)

    if args.exp_dirs:
        run_multi(args.exp_dirs, args.delta, args.significance, args.fig_dir)
    elif args.exp_dir:
        if not os.path.isdir(args.exp_dir):
            print(f"Error: experiment directory not found: {args.exp_dir}")
            sys.exit(1)
        run_single(args.exp_dir, args.delta, args.significance, args.fig_dir)
    else:
        parser.error("Provide either --exp-dir or --exp-dirs")


if __name__ == '__main__':
    main()
