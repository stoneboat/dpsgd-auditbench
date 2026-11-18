#!/usr/bin/env python3
"""Measure coverage and false rejections of the corrected DP-SGD auditor."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, logsumexp
from scipy.stats import binom, binomtest

from background_suppression_check import simulate_scores
from whitebox_auditing.ndis_1d import ndis_eps_lb_all


def exact_score_epsilon(steps, q, clip, noise_std, delta):
    """Numerically invert the IN-to-OUT divergence of the finite-T score law."""
    counts = np.arange(steps + 1)
    shifts = counts * clip / (noise_std * math.sqrt(steps))
    log_weights = binom.logpmf(counts, steps, q)
    positive = (counts > 0) & np.isfinite(log_weights)
    log_coefficients = log_weights - 0.5 * shifts**2

    def divergence(epsilon):
        # Under OUT the standardized score is N(0, 1). The mixture-to-normal
        # likelihood ratio is increasing, so the maximizing set is one tail.
        def log_ratio_minus_epsilon(z):
            return logsumexp(log_coefficients + shifts * z) - epsilon

        # Any one positive-shift component can witness that the ratio exceeds exp(epsilon).
        upper = np.min((epsilon - log_coefficients[positive]) / shifts[positive]) + 1
        threshold = brentq(log_ratio_minus_epsilon, 0, upper)
        log_in_tail = logsumexp(log_weights + log_ndtr(shifts - threshold))
        log_scaled_out_tail = epsilon + log_ndtr(-threshold)
        return math.exp(log_in_tail) * max(0.0, -math.expm1(log_scaled_out_tail - log_in_tail))

    if divergence(0) <= delta:
        return 0.0

    upper = 8.0
    while divergence(upper) > delta:
        upper *= 2
    return brentq(lambda epsilon: divergence(epsilon) - delta, 0, upper)


def run_replicate(seed, *, args, true_params):
    rng = np.random.default_rng(seed)
    in_scores = simulate_scores(args.n_in, True, args.steps, args.noise_std, args.q, args.clip, rng)
    out_scores = simulate_scores(args.n_out, False, args.steps, args.noise_std, args.q, args.clip, rng)

    result = ndis_eps_lb_all(
        in_scores, out_scores, args.delta,
        alpha=0.05, n_bootstrap=args.n_bootstrap, rng=np.random.default_rng(2),
        methods=("bootstrap_ellipsoid",), gaussian_correction_r=args.correction,
    )["bootstrap_ellipsoid"]

    offset = true_params - result["theta_hat"]
    squared_distance = offset @ np.linalg.solve(result["Sigma"], offset)
    return squared_distance <= result["r2"], result["eps_lb_corrected"]


def report_rate(label, count, reps):
    interval = binomtest(count, reps).proportion_ci(method="wilson")
    print(f"{label}: {count}/{reps} ({count/reps:.2%}, 95% Wilson CI "
          f"{interval.low:.2%} to {interval.high:.2%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults match the saved DP-SGD epsilon=8 run and the paper's correction.
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--q", type=float, default=0.08192)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=2.5732421875)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--correction", type=float, default=3.30e-5,
                        help="Certified r for the chosen Model 1 parameters.")
    parser.add_argument("--n-in", type=int, default=2512)
    parser.add_argument("--n-out", type=int, default=2488)
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if (not 0 < args.q <= 1 or not 0 < args.delta < 1 or args.steps < 1
            or not math.isfinite(args.clip) or args.clip <= 0
            or not math.isfinite(args.noise_std) or args.noise_std <= 0
            or min(args.n_in, args.n_out, args.n_bootstrap) < 2
            or min(args.reps, args.workers) < 1 or args.seed < 0):
        parser.error("Invalid Model 1 parameters or replicate counts.")

    # Bound the error for this particular (T, q, C, noise_std).
    noise_multiplier = args.noise_std / args.clip
    skew = args.q * (1 - args.q) * abs(1 - 2 * args.q)
    skew /= 6 * noise_multiplier**3 * math.sqrt(2 * math.pi * args.steps)
    remainder = args.q * (1 - args.q) / (12 * math.pi * noise_multiplier**4 * args.steps)
    if not math.isfinite(args.correction) or args.correction < 2 * min(1, skew + remainder):
        parser.error("The supplied correction is smaller than the certified bound for this setting.")

    true_params = np.array([
        args.clip * math.sqrt(args.steps) * args.q,
        math.sqrt(args.noise_std**2 + args.clip**2 * args.q * (1 - args.q)),
        0.0,
        args.noise_std,
    ])
    exact_epsilon = exact_score_epsilon(args.steps, args.q, args.clip, args.noise_std, args.delta)
    # Round upward at the precision displayed for the null privacy claim.
    null_epsilon = math.ceil(exact_epsilon * 1e6) / 1e6
    print(f"T={args.steps}, q={args.q}, C={args.clip}, noise std={args.noise_std}, "
          f"n_in={args.n_in}, n_out={args.n_out}, delta={args.delta}, r={args.correction}")
    print(f"IN-to-OUT Model 1 score epsilon = {exact_epsilon:.9f}; "
          f"testing the valid null epsilon={null_epsilon:.6f}", flush=True)

    seeds = np.random.SeedSequence(args.seed).spawn(args.reps)
    run = partial(run_replicate, args=args, true_params=true_params)
    covered = false_rejections = above_eight = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (contains_truth, lower_bound) in enumerate(pool.map(run, seeds), start=1):
            if not math.isfinite(lower_bound):
                raise RuntimeError(f"Replicate {i} returned a nonfinite lower bound.")
            covered += int(contains_truth)
            false_rejections += lower_bound > null_epsilon
            above_eight += lower_bound > 8
            if i % 25 == 0 or i == args.reps:
                print(f"Completed {i}/{args.reps}", flush=True)

    report_rate("Bootstrap ellipsoid coverage", covered, args.reps)
    report_rate("Corrected false rejections at the Model 1 score epsilon", false_rejections, args.reps)
    report_rate("Corrected lower bounds above epsilon=8", above_eight, args.reps)


if __name__ == "__main__":
    main()
