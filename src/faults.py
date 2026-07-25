"""Gaussian-preserving implementation faults for the white-box DP-SGD audit.

Each fault below is a realistic DP-SGD bug that changes the *moments* of the
canary-score channel while leaving it exactly Gaussian. Under Model 1 the score
is

    S_T = (1/sqrt(T)) sum_t [ B_t C 1{in} + sigma_t ],   sigma_t ~ N(0, (sigma * C)^2),

so a fault that only perturbs (sigma, q, C) keeps both score populations Gaussian:

    S^(0) ~ N(0, (sigma*C)^2),   S^(1) ~ N(sqrt(T) q C, (sigma*C)^2 + q(1-q)C^2).

The auditor never reads `sigma` -- it fits (mu_0, s_0, mu_1, s_1) from the scores --
so these faults are visible to it even though the transcript still "looks"
like a correct DP-SGD run.

Each fault reports both the *claimed* epsilon (what the accountant advertises,
i.e. what a deployment would publish) and the *true* epsilon of the mechanism
that actually ran. A valid audit must satisfy eps_lb <= eps_true; a useful one
should also satisfy eps_lb > eps_claimed, which is the violation being detected.

Faults that are NOT here on purpose:
  * bugs in the example -> gradient -> per-sample-clip pipeline (e.g. clipping
    per augmentation instead of per example). Dirac canaries are injected into
    `summed_grad` *after* clipping, so this audit is blind to them by
    construction. That is a threat-model limitation, not a Gaussianity one.
  * noise reuse across steps, per-parameter-group noise multipliers,
    data-dependent noise. Those break the CLT or cross-canary homogeneity and
    are the non-Gaussian tier; they belong in the false-violation-rate study,
    not here.
"""

from __future__ import annotations

from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier

FAULTS = ("none", "clip_mismatch", "sigma_stale", "q_mismatch")

DEFAULT_STRENGTH = {
    "none": 1.0,
    "clip_mismatch": 0.5,   # noise calibrated to C' = C/2 while clipping at C
    "sigma_stale": 0.5,     # budget calibrated for T/2 steps, T steps actually run
    "q_mismatch": 2.0,      # realized canary sampling rate is 2x the accounted one
}


def epsilon_of(noise_multiplier, sample_rate, steps, delta, accountant="prv"):
    """Epsilon of `steps` compositions of the subsampled Gaussian mechanism."""
    acct = create_accountant(mechanism=accountant)
    acct.history = [(float(noise_multiplier), float(sample_rate), int(steps))]
    return float(acct.get_epsilon(delta=delta))


def apply_fault(
    fault,
    strength,
    *,
    optimizer,
    sample_rate,
    steps,
    target_epsilon,
    target_delta,
    canary_prob,
    accountant="prv",
    logger=None,
):
    """Mutate `optimizer` / the canary schedule in place to plant `fault`.

    Returns a dict describing the planted fault, including the claimed and the
    true epsilon, which gets written into hparams.json for the audit table.
    """
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; choose from {FAULTS}")
    if strength is None:
        strength = DEFAULT_STRENGTH[fault]

    sigma_claimed = float(optimizer.noise_multiplier)
    q_claimed = float(canary_prob)
    eps_claimed = epsilon_of(sigma_claimed, sample_rate, steps, target_delta, accountant)

    sigma_used, q_used = sigma_claimed, q_claimed
    description = "correctly implemented DP-SGD"

    if fault == "clip_mismatch":
        # The clipping bound was tightened to C but the noise calibration still
        # uses the old, smaller sensitivity C' = strength * C. Opacus draws
        # noise with std = noise_multiplier * max_grad_norm, so scaling the
        # multiplier is exactly equivalent to calibrating noise to C'.
        sigma_used = sigma_claimed * strength
        description = (
            f"noise calibrated to sensitivity C'={strength}*C while clipping at C "
            f"(effective noise multiplier {sigma_used:.4f} vs claimed {sigma_claimed:.4f})"
        )

    elif fault == "sigma_stale":
        # The step budget was changed (e.g. --target-steps doubled) but the
        # noise multiplier was not recalibrated, so sigma still corresponds to the
        # old, shorter composition.
        steps_calibrated = max(1, int(round(strength * steps)))
        sigma_used = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=steps_calibrated,
            accountant=accountant,
        )
        description = (
            f"noise multiplier calibrated for T={steps_calibrated} steps but "
            f"T={steps} steps were run "
            f"(effective noise multiplier {sigma_used:.4f} vs claimed {sigma_claimed:.4f})"
        )

    elif fault == "q_mismatch":
        # The accountant was given the wrong sample rate (wrong dataset size,
        # dropped last batch, or a filter applied after the rate was computed),
        # so the realized per-step participation rate is higher than accounted.
        q_used = q_claimed * strength
        description = (
            f"realized canary sampling rate {q_used:.5f} vs accounted "
            f"{q_claimed:.5f} ({strength}x)"
        )

    # The true mechanism: whatever actually ran.
    sample_rate_true = sample_rate * (strength if fault == "q_mismatch" else 1.0)
    eps_true = epsilon_of(sigma_used, sample_rate_true, steps, target_delta, accountant)

    optimizer.noise_multiplier = sigma_used

    info = {
        "fault": fault,
        "fault_strength": float(strength),
        "fault_description": description,
        "noise_multiplier_claimed": sigma_claimed,
        "noise_multiplier_used": float(sigma_used),
        "canary_prob_claimed": q_claimed,
        "canary_prob_used": float(q_used),
        "sample_rate_claimed": float(sample_rate),
        "sample_rate_true": float(sample_rate_true),
        "eps_claimed": eps_claimed,
        "eps_true": eps_true,
        "accountant": accountant,
    }

    if logger is not None:
        logger.info(f"[fault] {fault}: {description}")
        logger.info(
            f"[fault] eps_claimed={eps_claimed:.4f}  eps_true={eps_true:.4f}  "
            f"(audit must satisfy eps_lb <= eps_true; detection means eps_lb > eps_claimed)"
        )

    return info