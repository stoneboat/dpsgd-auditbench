#!/usr/bin/env python3
"""Is the audit statistic the same with and without the background term?

Background suppression says the real records contribute nothing at a canary
coordinate, so the score pair is exactly the Model 1 pair the calibration
predicts. That is testable without retraining: the saved scores ARE the real
audit statistic, so simulate the background-free pair at the same
(sigma, C, T, q, n_in, n_out) and check the two agree.

Scores are GENERATED step by step from the background-free mechanism, the way
src/sgd/ndis_simulation.py does it -- not drawn from a fitted Gaussian:

      per step t:   z_t = N(0, tau^2) + C * Bernoulli(q)   [Bernoulli only if IN]
      score       = sum_t z_t / sqrt(T)                    tau = sigma * C

The generative model has no h_{t,z} term at all, so the background is absent by
construction; that is the point of the comparison. Summing the T Gaussian noise
draws is exactly N(0, T*tau^2) and the injection count is exactly Binom(T, q),
so the default collapses those two closed forms (an identity, not a CLT step).
--per-step runs the literal T-step loop instead, as a cross-check.

Reported per run: within-world moments, mu_GDP = (mu_in-mu_out)/sigma, and every
audit epsilon, each for the real scores and for `--reps` simulated replicates.
If the real numbers sit inside the simulated spread, the background changes the
audit by less than sampling noise, which is the claim.

NOTE on sd: compare WITHIN-world sds to tau. The pooled sd over in+out is
inflated by the mean separation (p(1-p)*mu_sep^2) and is not comparable to tau.

Usage:
  python scripts/background_suppression_check.py --exp-dirs ./data/run_eps1 ./data/run_eps8
  python scripts/background_suppression_check.py --exp-dirs ./data/*mislabeled* --reps 200
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.stats import anderson, ks_2samp

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(os.path.join(project_dir, 'src'))
sys.path.append(script_dir)

from whitebox_auditing.ndis_1d import ndis_eps_lb_all, _ndis_eps_from_moments  # noqa: E402
from auditing import CanaryScoreAuditor  # noqa: E402


def latest_epoch(exp_dir, prefix='in_scores_ndis_'):
    eps = sorted(int(f.split('_')[-1].replace('.csv', '')) for f in os.listdir(exp_dir)
                 if f.startswith(prefix) and f.endswith('.csv'))
    return eps[-1] if eps else None


def resolve_sum(exp_dir, side, epoch):
    for kind in ('sum', 'optimal'):
        p = os.path.join(exp_dir, f'{side}_scores_{kind}_{epoch:06d}.csv')
        if os.path.isfile(p):
            return p
    return None


def simulate_scores(n, inject, T, tau, q, C, rng, *, per_step=False,
                    deterministic_inject=False, mu_inject=0.0):
    """Background-free Model 1 scores, NDIS scale (raw sum / sqrt(T)).

    per_step=True runs the literal T-step loop: z_t = N(0, tau^2) + C*Bern(q).
    The default uses the two exact closed forms for that same law -- the noise
    sum is N(0, T*tau^2) and the injection count is Binom(T, q) -- which is an
    identity, not an approximation, and avoids an (n, T) array.
    deterministic_inject: DP-FTRL/MF inject once at a fixed leaf, so the IN mean
    shift is exactly mu_inject with no count randomness.
    """
    if per_step:
        z = rng.normal(0.0, tau, size=(n, T))
        if inject and not deterministic_inject:
            z += C * (rng.random((n, T)) < q)
        s = z.sum(axis=1) / math.sqrt(T)
        return s + mu_inject if (inject and deterministic_inject) else s
    s = rng.normal(0.0, tau * math.sqrt(T), size=n)
    if inject:
        s = s + (mu_inject * math.sqrt(T) if deterministic_inject
                 else C * rng.binomial(T, q, size=n))
    return s / math.sqrt(T)


def audit_all(in_ndis, out_ndis, in_sum, out_sum, delta, alpha, n_bootstrap, rng):
    """Every audit statistic on one score pair."""
    out = {}
    out['mu_in'] = float(in_ndis.mean())
    out['mu_out'] = float(out_ndis.mean())
    out['sd_in'] = float(in_ndis.std(ddof=1))
    out['sd_out'] = float(out_ndis.std(ddof=1))
    pooled_sd = math.sqrt(0.5 * (out['sd_in'] ** 2 + out['sd_out'] ** 2))
    out['mu_gdp'] = (out['mu_in'] - out['mu_out']) / pooled_sd if pooled_sd > 0 else float('nan')
    out['eps_point'] = _ndis_eps_from_moments(out['mu_in'], out['sd_in'], out['mu_out'], out['sd_out'], delta)
    try:
        lb = ndis_eps_lb_all(in_ndis, out_ndis, delta=delta, alpha=alpha, n_bootstrap=n_bootstrap, pool_variance=False, rng=rng)
        out['eps_ndis_bonf'] = float(lb['parametric_bonferroni']['eps_lb'])
        out['eps_ndis_boot'] = float(lb['bootstrap_ellipsoid']['eps_lb'])
    except (ValueError, RuntimeError):
        out['eps_ndis_bonf'] = out['eps_ndis_boot'] = 0.0
    if in_sum is not None:
        a = CanaryScoreAuditor(in_sum, out_sum)
        out['eps_steinke'], _ = a._epsilon_one_run_all_thresholds(significance=alpha, delta=delta, one_sided=True, threshold=None, use_fdp=False)
        out['eps_fdp'], _ = a._epsilon_one_run_all_thresholds(significance=alpha, delta=delta, one_sided=True, threshold=None, use_fdp=True)
    else:
        out['eps_steinke'] = out['eps_fdp'] = float('nan')
    return out


KEYS = ('mu_in', 'mu_out', 'sd_in', 'sd_out', 'mu_gdp', 'eps_point', 'eps_ndis_bonf', 'eps_ndis_boot', 'eps_steinke', 'eps_fdp')


def check_one(exp_dir, args, rng):
    hp_path = os.path.join(exp_dir, 'hparams.json')
    if not os.path.isfile(hp_path):
        print(f"  skip {exp_dir}: no hparams.json")
        return None
    with open(hp_path) as f:
        hp = json.load(f)
    epoch = latest_epoch(exp_dir)
    if epoch is None:
        print(f"  skip {exp_dir}: no in_scores_ndis_*.csv")
        return None

    in_ndis = np.loadtxt(os.path.join(exp_dir, f'in_scores_ndis_{epoch:06d}.csv'), delimiter=',')
    out_ndis = np.loadtxt(os.path.join(exp_dir, f'out_scores_ndis_{epoch:06d}.csv'), delimiter=',')
    p_in, p_out = resolve_sum(exp_dir, 'in', epoch), resolve_sum(exp_dir, 'out', epoch)
    in_sum = np.loadtxt(p_in, delimiter=',') if p_in else None
    out_sum = np.loadtxt(p_out, delimiter=',') if p_out else None

    C = float(hp.get('max_grad_norm', 1.0))
    T = int(hp['target_steps'])
    eps_target = hp.get('epsilon')
    if 'noise_multiplier' in hp:              # DP-SGD
        tau = float(hp['noise_multiplier']) * C
        B = int(hp.get('logical_batch_size', 4096))
        N = int(hp.get('train_set_size', args.train_set_size))
        q = B / N
        mu_in_pred = q * math.sqrt(T) * C
        sd_in_pred = math.sqrt(tau ** 2 + q * (1 - q) * C ** 2)
        mech = 'DP-SGD'
        det_inject = False
    elif 'sigma_node' in hp:                  # DP-FTRL tree / MF
        from whitebox_auditing.tree_mechanism import num_levels
        tau = float(hp['sigma_node'])
        q = float('nan')
        L = num_levels(T) + 1
        mu_in_pred = C if hp.get('mechanism') == 'mf' else math.sqrt(L) * C
        sd_in_pred = tau                      # deterministic single-leaf injection
        mech = f"DP-FTRL/{hp.get('mechanism', 'tree')}"
        det_inject = True
        if hp.get('mechanism') == 'mf':
            from whitebox_auditing import matrix_factorization as mf
            Cmat = mf.sqrt_strategy(T, bands=int(hp.get('mf_bands') or 0))
            mu_in_pred = C * float(np.linalg.norm(Cmat[:, 0]))
    else:
        print(f"  skip {exp_dir}: hparams has neither noise_multiplier nor sigma_node")
        return None

    n_in, n_out = len(in_ndis), len(out_ndis)
    sqrtT = math.sqrt(T)
    real = audit_all(in_ndis, out_ndis, in_sum, out_sum, args.delta, args.alpha, args.n_bootstrap, rng)

    sim = lambda n, inject: simulate_scores(n, inject, T, tau, q, C, rng,
                                            per_step=args.per_step,
                                            deterministic_inject=det_inject,
                                            mu_inject=mu_in_pred)

    sims = {k: [] for k in KEYS}
    for _ in range(args.reps):
        si, so = sim(n_in, True), sim(n_out, False)
        r = audit_all(si, so, si * sqrtT, so * sqrtT, args.delta, args.alpha, args.n_bootstrap, rng)
        for k in KEYS:
            sims[k].append(r[k])
    sims = {k: np.array(v, dtype=float) for k, v in sims.items()}

    # Two-sample KS + Anderson-Darling: real scores vs one background-free draw, per world.
    sim_in_1, sim_out_1 = sim(n_in, True), sim(n_out, False)
    ks_in = ks_2samp(in_ndis, sim_in_1)
    ks_out = ks_2samp(out_ndis, sim_out_1)
    ad_real_in, ad_real_out = anderson(in_ndis).statistic, anderson(out_ndis).statistic
    ad_sim_in, ad_sim_out = anderson(sim_in_1).statistic, anderson(sim_out_1).statistic

    print("=" * 96)
    print(f"{exp_dir}")
    print(f"  {mech}  eps_target={eps_target}  T={T}  C={C:g}  tau={tau:.4f}  q={q:.6g}  m={n_in + n_out} ({n_in} IN / {n_out} OUT)")
    inj = '+ C at a fixed leaf' if det_inject else f'+ {C:g}*Bern({q:.4g})'
    print(f"  generative model (no background): per step N(0, {tau:.4f}^2) {inj};  score = sum_t/sqrt(T)")
    print(f"  -> predicted OUT N(0, {tau:.4f}^2)   IN N({mu_in_pred:.4f}, {sd_in_pred:.4f}^2)")
    print("-" * 96)
    print(f"  {'statistic':<16}{'real':>12}{'simulated mean':>16}{'sim sd':>10}{'z':>8}{'pct':>7}   verdict")
    for k in KEYS:
        s = sims[k]
        mu_s, sd_s = float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0
        z = (real[k] - mu_s) / sd_s if sd_s > 0 else float('nan')
        pct = 100.0 * float(np.mean(s <= real[k]))
        flag = 'ok' if (not np.isfinite(z) or abs(z) < 3) else 'OUTSIDE'
        print(f"  {k:<16}{real[k]:>12.4f}{mu_s:>16.4f}{sd_s:>10.4f}{z:>8.2f}{pct:>6.0f}%   {flag}")
    print("-" * 96)
    print(f"  within-world sd vs tau: OUT {real['sd_out']:.4f} / {tau:.4f} = {real['sd_out'] / tau:.4f}   "
          f"IN {real['sd_in']:.4f} / {sd_in_pred:.4f} = {real['sd_in'] / sd_in_pred:.4f}")
    excess = real['sd_out'] ** 2 - tau ** 2
    print(f"  implied background variance (OUT): {excess:+.4f} = {excess / tau ** 2:+.2%} of tau^2")
    print(f"  KS vs background-free draw: OUT D={ks_out.statistic:.4f} p={ks_out.pvalue:.3f}   "
          f"IN D={ks_in.statistic:.4f} p={ks_in.pvalue:.3f}")
    print(f"  Anderson-Darling A^2:  real OUT={ad_real_out:.3f} IN={ad_real_in:.3f}   "
          f"simulated OUT={ad_sim_out:.3f} IN={ad_sim_in:.3f}   (15% critical 0.576)")
    row = dict(exp_dir=exp_dir, mech=mech, eps_target=eps_target, T=T, tau=tau, q=q,
               mu_in_pred=mu_in_pred, sd_in_pred=sd_in_pred, n_in=n_in, n_out=n_out,
               bg_var_excess=excess, bg_var_frac=excess / tau ** 2,
               ks_out_D=ks_out.statistic, ks_out_p=ks_out.pvalue,
               ks_in_D=ks_in.statistic, ks_in_p=ks_in.pvalue,
               ad_real_out=ad_real_out, ad_real_in=ad_real_in,
               ad_sim_out=ad_sim_out, ad_sim_in=ad_sim_in)
    for k in KEYS:
        row[f'real_{k}'] = real[k]
        row[f'sim_{k}'] = float(sims[k].mean())
        row[f'sim_{k}_sd'] = float(sims[k].std(ddof=1)) if args.reps > 1 else 0.0
    return row


def main():
    ap = argparse.ArgumentParser(description='Compare the real audit statistic against a background-free simulation.')
    ap.add_argument('--exp-dirs', nargs='+', required=True)
    ap.add_argument('--delta', type=float, default=1e-5)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--reps', type=int, default=100, help='Simulated replicates per run.')
    ap.add_argument('--n-bootstrap', type=int, default=500, help='Bootstrap draws inside the ellipsoid CR.')
    ap.add_argument('--train-set-size', type=int, default=50000, help='N for q = B/N when hparams lacks it.')
    ap.add_argument('--per-step', action='store_true',
                    help='Generate each score by the literal T-step loop instead of the two '
                         'equivalent closed forms. Same law; slower, (reps, n, T) work.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='Optional CSV summary path.')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = [r for r in (check_one(d.rstrip('/'), args, rng) for d in args.exp_dirs) if r is not None]
    if not rows:
        sys.exit('no usable exp_dirs')

    print("=" * 96)
    print(f"{'eps':>6}{'mu_GDP real':>14}{'mu_GDP sim':>12}{'eps_pt real':>13}{'eps_pt sim':>12}{'bg var':>10}{'KS p (OUT)':>12}")
    for r in sorted(rows, key=lambda x: (x['eps_target'] or 0)):
        print(f"{r['eps_target']:>6g}{r['real_mu_gdp']:>14.4f}{r['sim_mu_gdp']:>12.4f}"
              f"{r['real_eps_point']:>13.4f}{r['sim_eps_point']:>12.4f}{r['bg_var_frac']:>9.1%}{r['ks_out_p']:>12.3f}")
    print("=" * 96)
    print("Background suppression holds on a run when: bg var ~ 0, KS p not small, and every")
    print("statistic's |z| < 3 (real value inside the background-free simulated spread).")

    if args.out:
        cols = list(rows[0].keys())
        with open(args.out, 'w') as f:
            f.write(','.join(cols) + '\n')
            for r in rows:
                f.write(','.join(str(r[c]) for c in cols) + '\n')
        print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
