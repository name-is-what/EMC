#!/usr/bin/env bash
# Scaling study: runtime and peak GPU memory.
#
#     bash scripts/run_scaling.sh nodes      # node scaling, 11 domains
#     bash scripts/run_scaling.sh domains    # domain scaling, 10k nodes
#     bash scripts/run_scaling.sh baseline   # optional GCAL reference runs
#     bash scripts/run_scaling.sh all        # all of the above
#
# Results are written to logs/scaling/.

set -uo pipefail
source "$(dirname "$0")/_common.sh"

MODE="${1:-all}"

run_scaling_case() {
    local log_name="$1"
    local description="$2"
    shift 2
    run_logged "logs/scaling/${log_name}.log" "$description" \
        python scaling.py --device "$DEVICE" --seed "$SEED" "$@"
}

run_node_scaling() {
    for num_nodes in 1000 2000 5000 10000 15000 20000 21000 25000 30000; do
        run_scaling_case "emc-nodes-${num_nodes}" "EMC node scaling: ${num_nodes} nodes, 11 domains" \
            --method emc --num_nodes "$num_nodes" --num_graphs 11 --test_epochs 3
    done
}

run_domain_scaling() {
    for num_graphs in 11 21 31 41 51 61 71 81; do
        run_scaling_case "emc-domains-${num_graphs}" "EMC domain scaling: 10000 nodes, ${num_graphs} domains" \
            --method emc --num_nodes 10000 --num_graphs "$num_graphs" --test_epochs 3
    done
}

run_baseline_scaling() {
    for num_nodes in 25000 30000; do
        run_scaling_case "gcal-nodes-${num_nodes}" "GCAL node scaling: ${num_nodes} nodes, 11 domains" \
            --method gcal --num_nodes "$num_nodes" --num_graphs 11 \
            --test_epochs 10 --inner_loop 20
    done
}

case "$MODE" in
    nodes)    run_node_scaling ;;
    domains)  run_domain_scaling ;;
    baseline) run_baseline_scaling ;;
    all)      run_node_scaling; run_domain_scaling; run_baseline_scaling ;;
    *)
        echo "Usage: bash scripts/run_scaling.sh {nodes|domains|baseline|all}" >&2
        exit 1
        ;;
esac
