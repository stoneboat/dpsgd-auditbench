#!/bin/bash
#SBATCH --job-name=dpsgd_auditbench
#SBATCH --account=gts-vzikas3
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_audit_%j.out
#SBATCH --error=logs/dp_audit_%j.err

# === Module and env setup ===
module load anaconda3
conda activate /storage/scratch1/6/aagrawal609/conda_envs/dpsgd_bench

# === Run training ===
python scripts/gen_scores_DP_whitebox.py \
    --epochs 200 \
    --ckpt-interval 20 \
    --canary-count 5000 \
    --pkeep 0.5 \
    --noise-multiplier 3.0 \
    --logical-batch-size 4096 \
    --max-physical-batch-size 128 \
    --aug-multiplicity 1 \
    --lr 4.0 \
    --data-dir ./data \
    --log-dir ./logs