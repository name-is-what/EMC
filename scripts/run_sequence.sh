#!/usr/bin/env bash
# Domain-order and cyclic-shift experiments on Twitch.
#
#     bash scripts/run_sequence.sh
#     METHOD=gcal bash scripts/run_sequence.sh
#
# Results are written to logs/sequence/; per-trial accuracy matrices are saved
# to results/sequence/ when --save_matrix is enabled.

set -uo pipefail
source "$(dirname "$0")/_common.sh"

METHOD="${METHOD:-emc}"
SAVE_DIR="results/sequence"

run_sequence_case() {
    local log_name="$1"
    local description="$2"
    shift 2
    run_logged "logs/sequence/${log_name}.log" "$description" \
        python sequence.py \
        --dataset twitch \
        --model gcn \
        --method "$METHOD" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --test_epochs 3 \
        --save_dir "$SAVE_DIR" \
        --save_matrix \
        "$@"
}

run_sequence_case "ordered" "Ordered Twitch domain sequence" \
    --sequence_name twitch_ordered

for seq_seed in 0 1; do
    run_sequence_case "shuffle-${seq_seed}" "Shuffled Twitch domain sequence (seed ${seq_seed})" \
        --shuffle_targets --seq_seed "$seq_seed" \
        --sequence_name "twitch_shuffle_${seq_seed}"
done

run_sequence_case "cycle-all" "Full Twitch domain cycle (two passes)" \
    --repeat_targets 2 --sequence_name twitch_cycle_all

run_sequence_case "cycle-abc" "Three-domain Twitch cycle: ENGB, ES, FR" \
    --target_seq ENGB,ES,FR --repeat_targets 2 --sequence_name twitch_cycle_abc

run_sequence_case "cycle-def" "Three-domain Twitch cycle: PTBR, RU, TW" \
    --target_seq PTBR,RU,TW --repeat_targets 2 --sequence_name twitch_cycle_def
