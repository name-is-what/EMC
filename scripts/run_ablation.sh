#!/usr/bin/env bash
# Ablation study: full EMC vs. its two variants.
#
#     full            EMC with condensed-memory replay (reference)
#     no_replay       removes replay of the condensed memory
#     no_condensation replaces condensed memories with raw target graphs
#
#     bash scripts/run_ablation.sh
#     ABLATIONS="no_replay" bash scripts/run_ablation.sh
#
# Results are written to logs/ablation/<variant>-<dataset>-<model>.log.

set -uo pipefail
source "$(dirname "$0")/_common.sh"

ABLATIONS="${ABLATIONS:-full no_replay no_condensation}"

for ablation in $ABLATIONS; do
    echo "=== Running ablation: $ablation ==="
    main_sweep ablation "${ablation}-" --ablation "$ablation"
done
