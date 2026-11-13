#!/bin/bash
#SBATCH --job-name=dpsgd_lr_sweep
#SBATCH --account=gts-vzikas3
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_audit_lr_%A_%a.out
#SBATCH --error=logs/dp_audit_lr_%A_%a.err
#SBATCH --array=1-4

# Learning-rate ablation for the white-box DP-SGD audit.
#
# Only --lr changes across arms. Everything else -- eps=8, T=2500, B=4096,
# C=1.0, K=16, EMA 0.9999, m=5000, pkeep=0.5 -- matches the existing baseline
# run (job 7901948_3) at
#   ./data/mislabeled-canaries-277640617625635548156432291773894477542-5000-0.5-cifar10
# which is the LR=4.0 point of this sweep and is NOT re-run here.
#
# DATABASE_SEED is that run's seed, so every arm shares its canary coordinates
# and inclusion mask; LR is the only difference. --exp-tag keeps each arm in its
# own exp_dir instead of overwriting the baseline.
#
#   task  lr     exp_dir suffix
#   -     4.0    (none -- existing baseline run)
#   1     1.0    -lr1.0
#   2     2.0    -lr2.0
#   3     8.0    -lr8.0
#   4     16.0   -lr16.0
#
# LR does not enter the privacy accountant, so sigma, q and eps_true are
# identical across all arms; only utility and audit power should move.

# === Module and env setup ===
module load anaconda3
conda activate /storage/scratch1/6/aagrawal609/conda_envs/audit

# === Shared configuration ===
# Seed of the existing LR=4.0 run (job 7901948_3), so canary placement and the
# inclusion mask match it exactly.
DATABASE_SEED=277640617625635548156432291773894477542
TARGET_STEPS=2500
EPS=8

LRS=(1.0 2.0 8.0 16.0)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

echo "=== array task ${SLURM_ARRAY_TASK_ID}: lr=${LR} ==="
echo "epsilon=${EPS}, target steps=${TARGET_STEPS}, seed=${DATABASE_SEED}"

# === Run training (Sander/Mahloujifar recipe: K=16, EMA 0.9999) ===
python scripts/gen_scores_DP_whitebox.py \
    --lr ${LR} \
    --exp-tag lr${LR} \
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
    --max-grad-norm 1.0 \
    --data-dir ./data \
    --log-dir ./logs \
    --log-file logs/whitebox_eps8_lr${LR}_${SLURM_ARRAY_JOB_ID}.log
