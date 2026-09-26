#!/usr/bin/env python3
"""Q-Q plot: real OUT canary scores / tau for each run vs the background-free prediction N(0, 1).

Usage:
  python scripts/plot_bg_suppression.py --csv bg_suppression.csv
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # categorical slots 1-4


def load(row):
    d = row['exp_dir']
    epochs = sorted(int(f.split('_')[-1][:-4]) for f in os.listdir(d) if f.startswith('out_scores_ndis_') and f.endswith('.csv'))
    out = np.loadtxt(os.path.join(d, f'out_scores_ndis_{epochs[-1]:06d}.csv'), delimiter=',')
    return float(row['eps_target']), out / float(row['tau'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='bg_suppression.csv', help='output of background_suppression_check.py --out')
    ap.add_argument('--out', default='fig/bg_suppression')
    args = ap.parse_args()

    runs = sorted(load(r) for r in csv.DictReader(open(args.csv)))
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    for (eps, z), c in zip(runs, COLORS):
        theo = norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))  # N(0,1) plotting positions
        ax.plot(theo, np.sort(z), lw=1.5, color=c, label=f'$\\varepsilon={eps:g}$')
    ax.plot([-4, 4], [-4, 4], color='black', lw=1, ls='--', label='$y=x$')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_xlabel('$\\mathcal{N}(0,1)$ quantile')
    ax.set_ylabel('OUT score / $\\tau$ quantile')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300)
    print(f'wrote {args.out}.pdf/.png')


if __name__ == '__main__':
    main()
