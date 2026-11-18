#!/usr/bin/env bash
# Run inside the allocated SCC GPU desktop terminal.
set -euo pipefail

module load miniconda
export W="/projectnb/rlvn/students/$USER"
export PROJECT="$W/ec518"
export PIP_CACHE_DIR="$W/.pip_cache"
export HF_HOME="$W/.hf_cache"
export TORCH_HOME="$W/.torch_cache"
conda activate "$W/envs/ec518"
cd "$PROJECT/ec518-project"

DATA_ROOT="/projectnb/rlvn/students/zzongyu/ec518/pilot"
FEATURE_ROOT="$PROJECT/features-pilot-sky0"
RUN_ROOT="$PROJECT/runs-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT" > "$PROJECT/latest-training-run.txt"
exec > >(tee "$RUN_ROOT/training.log") 2>&1
printf 'Results directory: %s\n' "$RUN_ROOT"

python -u smoke_variants.py \
    --data-root "$DATA_ROOT" --feature-root "$FEATURE_ROOT" --device cuda:0

for variant in rgb vfm_implicit; do
    python -u train.py --variant "$variant" --seed 0 --val-seed 7 \
        --data-root "$DATA_ROOT" --feature-root "$FEATURE_ROOT" \
        --output-root "$RUN_ROOT/smoke" --device cuda:0 \
        --epochs 2 --batch-size 64 --workers 4
done

for variant in rgb vfm_implicit; do
    python -u train.py --variant "$variant" --seed 0 --val-seed 7 \
        --data-root "$DATA_ROOT" --feature-root "$FEATURE_ROOT" \
        --output-root "$RUN_ROOT" --device cuda:0 \
        --epochs 20 --batch-size 64 --workers 4
done

python plot_runs.py "$RUN_ROOT/rgb_seed0" "$RUN_ROOT/vfm_implicit_seed0" \
    --output "$RUN_ROOT/figures"

for variant in rgb vfm_implicit; do
    python -u check_policy.py "$RUN_ROOT/${variant}_seed0/${variant}_seed0.pt" \
        --data-root "$DATA_ROOT" --feature-root "$FEATURE_ROOT" \
        --device cuda:0 --frames 20
done

printf '\nTraining and consistency checks completed: %s\n' "$RUN_ROOT"
