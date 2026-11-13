#!/usr/bin/env python3
"""KS and Anderson-Darling normality of the canary-score populations.

Reports, for each experiment directory, the two goodness-of-fit statistics on
the IN and OUT score samples separately:

  KS  = sup_x |F_n(x) - Phi(x)|  after standardizing by the fitted moments.
        Bulk-sensitive; this is exactly the quantity Theorem 1 / Corollary 1
        bound, so it is the "matches-the-theory" number.
  A^2 = Anderson-Darling, the whole-curve squared gap weighted by
        1/[Phi(1-Phi)]. That weight diverges in the tails, so A^2 is the more
        privacy-relevant statistic -- small-delta epsilon lives in the tails.

IN and OUT are tested separately and never pooled: under Model 1 each is
Gaussian on its own, but their union is a two-component mixture and would fail
normality even for a perfect implementation.

Both p-values are valid under fitted parameters. A^2 uses scipy's
estimated-parameter critical values; KS uses the Lilliefors parametric
bootstrap (see canary_score_diagnostics.lilliefors_ks -- scipy's kstest
p-value is anti-conservative here and must not be used).

Usage
-----
  python scripts/score_normality_table.py --exp-dirs ./data/run_a ./data/run_b
  python scripts/score_normality_table.py --exp-dirs ./data/*fault* --epoch 200
"""

import os
import sys
import json
import argparse

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(script_dir)

from canary_score_diagnostics import gaussianity_tests  # noqa: E402


def find_epoch(exp_dir, epoch=None):
    """Resolve the score epoch to audit: `epoch` if given, else the latest."""
    epochs = sorted(
        int(f.split('_')[-1].replace('.csv', ''))
        for f in os.listdir(exp_dir)
        if f.startswith('in_scores_ndis_') and f.endswith('.csv')
    )
    if not epochs:
        return None
    if epoch is None:
        return epochs[-1]
    return epoch if epoch in epochs else None


def load_hparams(exp_dir):
    path = os.path.join(exp_dir, 'hparams.json')
    return json.load(open(path)) if os.path.isfile(path) else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp-dirs', type=str, nargs='+', required=True)
    ap.add_argument('--epoch', type=int, default=None,
                    help='Score epoch to test. Default: latest available per run. '
                         'Pass an explicit epoch to compare runs at matched T.')
    ap.add_argument('--out-csv', type=str, default=None)
    args = ap.parse_args()

    rows = []
    for exp_dir in args.exp_dirs:
        if not os.path.isdir(exp_dir):
            print(f"Warning: {exp_dir} not found, skipping")
            continue
        ep = find_epoch(exp_dir, args.epoch)
        if ep is None:
            print(f"Warning: no in_scores_ndis_*.csv"
                  f"{f' at epoch {args.epoch}' if args.epoch else ''}"
                  f" in {exp_dir}, skipping")
            continue

        hp = load_hparams(exp_dir)
        name = hp.get('fault', 'none')
        if name == 'none':
            name = 'none (control)'

        in_scores = np.loadtxt(os.path.join(exp_dir, f'in_scores_ndis_{ep:06d}.csv'),
                               delimiter=',')
        out_scores = np.loadtxt(os.path.join(exp_dir, f'out_scores_ndis_{ep:06d}.csv'),
                                delimiter=',')

        for side, sc in (('OUT', out_scores), ('IN', in_scores)):
            res = gaussianity_tests(sc, label=f'{name}/{side}')
            rows.append({
                'fault': name, 'side': side, 'epoch': ep, 'n': res['N'],
                'sigma_hat': res['std'], 'skew': res['skew'],
                'exc_kurt': res['excess_kurtosis'],
                'ks_D': res['ks_D'], 'ks_p': res['ks_p'],
                'A2': res['anderson_stat'],
                'A2_crit5': res['anderson_critical_5pct'],
                'A2_reject': res['anderson_reject_5pct'],
            })

    if not rows:
        sys.exit("No runs produced results.")

    hdr = (f"{'fault':16s} {'side':4s} {'ep':>4s} {'n':>5s} {'sigma':>7s} "
           f"{'skew':>7s} {'exc_kurt':>8s} {'KS_D':>7s} {'KS_p':>6s} "
           f"{'A^2':>6s} {'A^2_5%':>6s}  normal?")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        verdict = 'NON-GAUSSIAN' if (r['A2_reject'] or r['ks_p'] < 0.05) else 'ok'
        print(f"{r['fault']:16s} {r['side']:4s} {r['epoch']:4d} {r['n']:5d} "
              f"{r['sigma_hat']:7.4f} {r['skew']:7.4f} {r['exc_kurt']:8.4f} "
              f"{r['ks_D']:7.4f} {r['ks_p']:6.3f} {r['A2']:6.3f} "
              f"{r['A2_crit5']:6.3f}  {verdict}")

    if args.out_csv:
        cols = list(rows[0].keys())
        with open(args.out_csv, 'w') as f:
            f.write(','.join(cols) + '\n')
            for r in rows:
                f.write(','.join(str(r[c]) for c in cols) + '\n')
        print(f"\nSaved to: {args.out_csv}")


if __name__ == '__main__':
    main()