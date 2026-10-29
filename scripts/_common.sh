#!/usr/bin/env bash
# Shared helpers for the experiment launchers.
#
# All launchers source this file, so they run from the repository root and use
# the same logging format:
#
#     logs/<experiment>/<configuration>.log
#
# Override the defaults through environment variables, e.g.
#     DEVICE=1 SEED=1 NUM_TRIAL=3 bash scripts/run_main.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-0}"
SEED="${SEED:-0}"
NUM_TRIAL="${NUM_TRIAL:-5}"

mkdir -p logs checkpoints

run_logged() {
    # run_logged <log_file> <description> <command...>
    local log_file="$1"
    local description="$2"
    shift 2

    mkdir -p "$(dirname "$log_file")"
    {
        echo "=============================================="
        echo "Experiment: $description"
        printf 'Command: '
        printf '%q ' "$@"
        printf '\nStarted: %s\n' "$(date -Is)"
        echo "=============================================="
    } | tee -a "$log_file"

    if "$@" 2>&1 | tee -a "$log_file"; then
        echo "SUCCESS: $description" | tee -a "$log_file"
    else
        echo "FAILED: $description" | tee -a "$log_file"
    fi
}

run_case() {
    # run_case <script> <log_dir> <log_prefix> <dataset> <model> [extra args...]
    local script="$1"
    local log_dir="$2"
    local log_prefix="$3"
    local dataset="$4"
    local model="$5"
    shift 5

    run_logged "logs/${log_dir}/${log_prefix}${dataset}-${model}.log" \
        "${script} ${dataset}/${model}" \
        python "$script" \
        --dataset "$dataset" \
        --model "$model" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --num_trial "$NUM_TRIAL" \
        "$@"
}

# Main-table configurations (dataset, backbone, dataset-specific arguments).
# These are the settings used for the paper's main table.
main_sweep() {
    local log_dir="${1:-main}"
    local log_prefix="${2:-}"
    shift 2 || true

    while read -r dataset model extra; do
        [[ -z "${dataset}" ]] && continue
        # shellcheck disable=SC2086
        run_case main.py "$log_dir" "$log_prefix" "$dataset" "$model" $extra "$@"
    done <<'EOF'
twitch gcn --test_epochs 3
twitch sage --test_epochs 3
twitch gat --test_epochs 3
twitch gin --test_epochs 3
fb100 gcn --test_epochs 20 --dropout 0.5
fb100 sage --test_epochs 20 --dropout 0.1
fb100 gat --test_epochs 100 --dropout 0.6 --gat_heads 2 --hidden_channels 128
fb100 gin --test_epochs 20 --dropout 0.5
ogbn-arxiv gcn --test_epochs 3 --dropout 0.2
ogbn-arxiv sage --test_epochs 3 --dropout 0.3
ogbn-arxiv gat --test_epochs 3 --dropout 0.1
ogbn-arxiv gin --test_epochs 3 --dropout 0.1
elliptic gcn --test_epochs 5 --dropout 0.3
elliptic sage --test_epochs 3 --dropout 0.2
elliptic gat --test_epochs 3 --dropout 0.9
elliptic gin --test_epochs 5 --dropout 0.3
EOF
}
