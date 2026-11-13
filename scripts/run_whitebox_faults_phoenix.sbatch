#!/bin/bash
#SBATCH --job-name=dpsgd_faults
#SBATCH --account=gts-vzikas3
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_fault_%A_%a.out
#SBATCH --error=logs/dp_fault_%A_%a.err
#SBATCH --array=1-3

# Gaussian-preserving fault injection for the white-box DP-SGD audit.
#
# Tasks 1-3 plant one realistic implementation bug each. The control
# (fault=none) is NOT re-run here: job 7901948_3 already produced it at
# ./data/mislabeled-canaries-277640617625635548156432291773894477542-5000-0.5-cifar10
# (eps=8, T=2500, m=5000, sigma=2.6245, 69.40% EMA acc, ~11.6 GPU-hours).
# DATABASE_SEED below is that run's seed, so these arms share its canary
# coordinates and inclusion mask -- the fault is the only difference. The
# --fault tag in exp_dir keeps them from overwriting the control.
#
# Every run advertises eps_claimed = 7.98. The true epsilon of the mechanism
# that actually ran is written to <exp_dir>/hparams.json as `eps_true`:
#
#   task  fault            z(used)   q(used)   eps_claimed   eps_true
#   -     none (existing)  2.6245    0.0833    7.98           7.98
#   1     clip_mismatch    1.3122    0.0833    7.98          21.98
#   2     sigma_stale      1.9287    0.0833    7.98          12.12
#   3     q_mismatch       2.6245    0.1667    7.98          18.88
#
# A VALID audit satisfies eps_lb <= eps_true. A run DETECTS the violation when
# eps_lb > eps_claimed.

# === Module and env setup ===
module load anaconda3
conda activate /storage/scratch1/6/aagrawal609/conda_envs/audit

# === Shared configuration ===
# Seed of the existing control run (job 7901948_3), so canary placement and the
# inclusion mask match it exactly.
DATABASE_SEED=277640617625635548156432291773894477542
TARGET_STEPS=2500        # paper setting; drop to 500 for a fast smoke test
EPS=8

FAULTS=(none clip_mismatch sigma_stale q_mismatch)
STRENGTHS=(1.0 0.5 0.5 2.0)

FAULT=${FAULTS[$SLURM_ARRAY_TASK_ID]}
STRENGTH=${STRENGTHS[$SLURM_ARRAY_TASK_ID]}

echo "=== array task ${SLURM_ARRAY_TASK_ID}: fault=${FAULT} strength=${STRENGTH} ==="
echo "claimed epsilon=${EPS}, target steps=${TARGET_STEPS}, seed=${DATABASE_SEED}"

# === Run training (Sander/Mahloujifar recipe: K=16, EMA 0.9999) ===
python scripts/gen_scores_DP_whitebox.py \
    --fault ${FAULT} \
    --fault-strength ${STRENGTH} \
    --database-seed ${DATABASE_SEED} \
    --target-steps ${TARGET_STEPS} \
    --ema-decay 0.9999 \
    --ckpt-interval 20 \
    --canary-count 5000 \
    --pkeep 0.5 \
    --epsilon ${EPS} \
    --logical-batch-size 4096 \
    --max-physical-batch-size 128 \
    --aug-multiplicity 16 \
    --lr 4.0 \
    --data-dir ./data \
    --log-dir ./logs