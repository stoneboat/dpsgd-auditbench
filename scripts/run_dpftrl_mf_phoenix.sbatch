#!/bin/bash
#SBATCH --job-name=dpftrl_mf
#SBATCH --account=gts-vzikas3
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dpftrl_mf_%A_%a.out
#SBATCH --error=logs/dpftrl_mf_%A_%a.err
#SBATCH --array=0-3

# MF-DP-FTRL audit on ScatterLinear. Answers reviewer v2K8: per-round observations
# are dependent (rows of B mix the same nu), so Model 1's independence fails, yet
# the aggregate score is a linear functional of a Gaussian and the pair is exact.
#
# Strategy C is the closed-form matrix square root of the prefix-sum matrix; no
# optimization to solve, no extra dependency. At T=128 sensitivity is 1.62 vs the
# tree's 2.83, so MF runs at 1.75x less noise for the same eps.
#
# All canaries sit at leaf 0 so ||C[:, t]|| is common and the pooled population is
# exactly Gaussian; --random-leaves reproduces the tree-style assignment and turns
# the in-population into a mixture (a class-E heterogeneity ablation).
#
# Compare against the tree arm at the same seed via run_dpftrl_scatter_phoenix.sbatch.

module load anaconda3
conda activate /storage/scratch1/6/aagrawal609/conda_envs/dpsgd_bench

EPSILONS=(1 2 4 8)
EPS=${EPSILONS[$SLURM_ARRAY_TASK_ID]}
DATABASE_SEED=277640617625635548156432291773894477542

echo "Running MF-DP-FTRL audit for epsilon: $EPS"
python scripts/gen_scores_DP_FTRL_scatter.py \
    --mechanism mf \
    --mf-bands 0 \
    --database-seed ${DATABASE_SEED} \
    --target-steps 128 \
    --logical-batch-size 380 \
    --max-physical-batch-size 390 \
    --aug-multiplicity 1 \
    --canary-count 5000 \
    --pkeep 0.5 \
    --epsilon ${EPS} \
    --lr 1.0 \
    --data-dir ./data \
    --log-dir ./logs
