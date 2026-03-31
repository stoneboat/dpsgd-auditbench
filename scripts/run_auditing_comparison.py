#!/usr/bin/env python3
"""
Run all 3 auditing methods on saved whitebox scores and plot comparison.

Methods:
  1. Steinke et al. 2023 (one-run, Theorem 5.2) — uses raw sum scores
  2. Mahloujifar et al. 2024 (f-DP) — uses raw sum scores
  3. NDIS (normal distribution indistinguishability spectrum) — uses normalized scores

Usage:
  python scripts/run_auditing_comparison.py --exp-dir ./data/mislabeled-canaries-<seed>-5000-0.5-cifar10
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)

from auditing import CanaryScoreAuditor
from whitebox_auditing.ndis_1d import ndis_eps_from_delta_1d_brentq, estimate_mean_variance


def main():
    parser = argparse.ArgumentParser(description='Run 3 auditing methods on whitebox scores')
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Experiment directory containing score files')
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--significance', type=float, default=0.05,
                        help='Significance level for paper methods (95%% confidence)')
    parser.add_argument('--noise-multiplier', type=float, default=3.0)
    parser.add_argument('--fig-dir', type=str, default=None,
                        help='Directory to save figures (default: <project>/fig)')
    parser.add_argument('--steps-per-epoch', type=int, default=None,
                        help='Steps per epoch (auto-detected from privacy_params if not set)')
    args = parser.parse_args()

    if args.fig_dir is None:
        args.fig_dir = os.path.join(project_dir, 'fig')
    os.makedirs(args.fig_dir, exist_ok=True)

    exp_dir = args.exp_dir
    if not os.path.isdir(exp_dir):
        print(f"Error: experiment directory not found: {exp_dir}")
        sys.exit(1)

    # Discover available epochs from score files
    epochs = sorted(set(
        int(f.split('_')[-1].replace('.csv', ''))
        for f in os.listdir(exp_dir)
        if f.startswith('in_scores_sum_') and f.endswith('.csv')
    ))

    if not epochs:
        print(f"Error: no in_scores_sum_*.csv files found in {exp_dir}")
        sys.exit(1)

    print(f"Found score files for epochs: {epochs}")

    # Compute theoretical upper bound
    # Need opacus for RDP accounting
    from opacus.accountants import RDPAccountant

    # Detect steps_per_epoch from privacy_params or use provided value
    # Load hparams if available
    hparams_path = os.path.join(exp_dir, 'hparams.json')
    if os.path.isfile(hparams_path):
        import json
        with open(hparams_path) as f:
            hparams = json.load(f)
        logical_batch_size = hparams.get('logical_batch_size', 4096)
        noise_multiplier = hparams.get('noise_multiplier', args.noise_multiplier)
        print(f"Loaded hparams: batch={logical_batch_size}, sigma={noise_multiplier}")
    else:
        noise_multiplier = args.noise_multiplier
        logical_batch_size = 4096

    # CIFAR-10 has 50000 samples, steps_per_epoch = ceil(50000 / batch_size)
    n_train = 50000
    if args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch
    else:
        steps_per_epoch = int(np.ceil(n_train / logical_batch_size))
    sample_rate = 1.0 / steps_per_epoch
    print(f"Steps per epoch: {steps_per_epoch}, sample_rate: {sample_rate:.6f}")

    # Run auditing for each epoch
    epoch_list = []
    upper_bounds = []
    steinke_bounds = []
    fdp_bounds = []
    ndis_bounds = []

    for epoch in epochs:
        # Load raw sum scores (for paper methods)
        in_sum_path = os.path.join(exp_dir, f'in_scores_sum_{epoch:06d}.csv')
        out_sum_path = os.path.join(exp_dir, f'out_scores_sum_{epoch:06d}.csv')
        # Load normalized scores (for NDIS)
        in_ndis_path = os.path.join(exp_dir, f'in_scores_ndis_{epoch:06d}.csv')
        out_ndis_path = os.path.join(exp_dir, f'out_scores_ndis_{epoch:06d}.csv')

        if not all(os.path.isfile(p) for p in [in_sum_path, out_sum_path, in_ndis_path, out_ndis_path]):
            print(f"  Epoch {epoch}: missing score files, skipping")
            continue

        in_sum = np.loadtxt(in_sum_path, delimiter=',')
        out_sum = np.loadtxt(out_sum_path, delimiter=',')
        in_ndis = np.loadtxt(in_ndis_path, delimiter=',')
        out_ndis = np.loadtxt(out_ndis_path, delimiter=',')

        # 1. Theoretical upper bound (RDP accounting)
        total_steps = steps_per_epoch * epoch
        accountant = RDPAccountant()
        accountant.history.append((noise_multiplier, sample_rate, total_steps))
        eps_upper = accountant.get_epsilon(delta=args.delta)

        # 2. Steinke et al. 2023 (one-run) — raw sum scores
        auditor = CanaryScoreAuditor(in_sum, out_sum)
        eps_steinke = auditor.epsilon_one_run(
            significance=args.significance, delta=args.delta
        )

        # 3. Mahloujifar et al. 2024 (f-DP) — raw sum scores
        eps_fdp = auditor.epsilon_one_run_fdp(
            significance=args.significance, delta=args.delta
        )

        # 4. NDIS — normalized scores
        stats = estimate_mean_variance(in_ndis, out_ndis)
        try:
            # NDIS requires sigma1 < sigma2
            if stats['out_std'] < stats['in_std']:
                eps_ndis = ndis_eps_from_delta_1d_brentq(
                    sigma1=stats['out_std'],
                    sigma2=stats['in_std'],
                    mu1=stats['out_mean'],
                    mu2=stats['in_mean'],
                    delta_target=args.delta,
                )
            else:
                eps_ndis = ndis_eps_from_delta_1d_brentq(
                    sigma1=stats['in_std'],
                    sigma2=stats['out_std'],
                    mu1=stats['in_mean'],
                    mu2=stats['out_mean'],
                    delta_target=args.delta,
                )
        except (ValueError, RuntimeError) as e:
            print(f"  Epoch {epoch}: NDIS failed ({e}), setting to 0")
            eps_ndis = 0.0

        epoch_list.append(epoch)
        upper_bounds.append(eps_upper)
        steinke_bounds.append(eps_steinke)
        fdp_bounds.append(eps_fdp)
        ndis_bounds.append(eps_ndis)

        print(f"  Epoch {epoch:4d}: upper={eps_upper:.3f}, "
              f"steinke={eps_steinke:.3f}, fdp={eps_fdp:.3f}, ndis={eps_ndis:.3f}")

    # Save results to CSV
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

    fig_path = os.path.join(args.fig_dir, 'privacy_bounds_comparison.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")


if __name__ == '__main__':
    main()