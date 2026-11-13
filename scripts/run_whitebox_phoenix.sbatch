#!/bin/bash
#SBATCH --job-name=dpsgd_auditbench
#SBATCH --account=gts-vzikas3
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_audit_ind%j.out
#SBATCH --error=logs/dp_audit_ind%j.err
#SBATCH --array=0

# === Module and env setup ===
module load anaconda3
conda activate /storage/scratch1/6/aagrawal609/conda_envs/audit

# === Epsilon values ===
EPSILONS=(8)
EPS=${EPSILONS[$SLURM_ARRAY_TASK_ID]}

echo "Running with target epsilon=${EPS} (array task ${SLURM_ARRAY_TASK_ID})"

# === Run training (Sander/Mahloujifar recipe: 2500 steps, K=16, EMA 0.9999) ===
python scripts/gen_scores_DP_whitebox.py \
    --target-steps 500 \
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
