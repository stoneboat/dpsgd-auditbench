#!/bin/bash
#SBATCH --job-name=ndis_ablation_T
#SBATCH --account=gts-vzikas3
#SBATCH --output=logs/ablation_T_%A_%a.out
#SBATCH --error=logs/ablation_T_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# 1. Environment
module load anaconda3/2023.03
source activate audit

# 2. Ablation grid: vary number of DP-SGD steps T at fixed (eps, batch).
#    Privacy accountant rescales sigma so each T hits the same target eps;
#    this isolates the effect of step count on utility + audit power.
T_VALUES=(500 1000 2500 5000)
CURRENT_T=${T_VALUES[$SLURM_ARRAY_TASK_ID]}
EPS=8.0

echo "=========================================================="
echo "DP-SGD Convergence Ablation: T=${CURRENT_T}, eps=${EPS}"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "=========================================================="

# 3. Run. Note: script lives under scripts/, not src/.
python scripts/gen_scores_DP_whitebox.py \
    --target-steps ${CURRENT_T} \
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

echo "Job completed for T = ${CURRENT_T}."