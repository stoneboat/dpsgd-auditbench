#!/bin/bash
#SBATCH --job-name=fixed_gaussianity_error
#SBATCH --account=gts-vzikas3
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=fig/fixed_gaussianity_error/slurm-%A_%a.out
#SBATCH --error=fig/fixed_gaussianity_error/slurm-%A_%a.err

set -e
cd "$SLURM_SUBMIT_DIR"
module load anaconda3
conda activate "${AUDIT_ENV:-/storage/scratch1/6/aagrawal609/conda_envs/audit}"
set -uo pipefail
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

data_dir="${AUDIT_DATA_DIR:-$PWD/data}"
result_dir="$PWD/fig/fixed_gaussianity_error/$SLURM_ARRAY_JOB_ID"
sgd8="$data_dir/mislabeled-canaries-277640617625635548156432291773894477542-5000-0.5-cifar10"
audit=(srun python -u "$PWD/scripts/run_auditing_comparison.py" --compare-tails --delta 1e-5 --significance 0.05)

case "$SLURM_ARRAY_TASK_ID" in
    0)
        "${audit[@]}" --fig-dir "$result_dir/epsilon" --exp-dirs \
            "$data_dir/mislabeled-canaries-194358092235300364301779860499483305667-5000-0.5-cifar10" \
            "$data_dir/mislabeled-canaries-33983531732666956501283061792610382617-5000-0.5-cifar10" \
            "$data_dir/mislabeled-canaries-150826909777158496980686348097484970563-5000-0.5-cifar10" \
            "$sgd8"
        ;;
    1)
        "${audit[@]}" --fig-dir "$result_dir/steps" --ablation-T --exp-dirs \
            "$data_dir/mislabeled-canaries-227413209713036813868017938866853468203-5000-0.5-cifar10" \
            "$data_dir/mislabeled-canaries-6361188182970055010550653208276212089-5000-0.5-cifar10" \
            "$data_dir/mislabeled-canaries-60281807569336015233236270129373243223-5000-0.5-cifar10" \
            "$data_dir/mislabeled-canaries-32554249179507031055760171971516307803-5000-0.5-cifar10" \
            "$sgd8"
        ;;
    2)
        "${audit[@]}" --fig-dir "$result_dir/canaries" --exp-dir "$sgd8" \
            --complexity --complexity-trials "${AUDIT_COMPLEXITY_TRIALS:-50}"
        ;;
    3)
        table_runs=(
            "$sgd8-fault_clip_mismatch" "$sgd8-fault_sigma_stale" "$sgd8-fault_q_mismatch"
            "$sgd8-adaptive" "$sgd8-lr2.0" "$sgd8-lr8.0" "$sgd8-lr16.0"
        )
        # Supply these paths for the remaining rows. The clip-10 input must already be unit-normalized.
        for name in AUDIT_BATCH1024 AUDIT_BATCH2048 AUDIT_CLIP10_NORMALIZED; do
            if [[ -n "${!name:-}" ]]; then
                table_runs+=("${!name}")
            else
                printf 'Set %s to include its table row.\n' "$name"
            fi
        done
        "${audit[@]}" --fig-dir "$result_dir/tables" --csv-only --exp-dirs "${table_runs[@]}"
        ;;
esac
