#!/usr/bin/env bash
# Main table: EMC on Twitch / FB100 / ogbn-arxiv / Elliptic with GCN, SAGE,
# GAT and GIN backbones.
#
#     bash scripts/run_main.sh
#     DEVICE=1 SEED=1 NUM_TRIAL=3 bash scripts/run_main.sh
#
# Results are written to logs/main/<dataset>-<model>.log.

set -uo pipefail
source "$(dirname "$0")/_common.sh"

main_sweep main ""
