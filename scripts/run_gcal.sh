#!/usr/bin/env bash
# Baseline comparison: GCAL on the configurations reported in the paper.
#
#     bash scripts/run_gcal.sh
#
# Results are written to logs/gcal/<dataset>-<model>.log.  Add other
# backbones by copying a line and changing --model.

set -uo pipefail
source "$(dirname "$0")/_common.sh"

run_case main_gcal.py gcal "" twitch gcn \
    --test_epochs 3 --train_epochs 100 \
    --entropy_weight 5 --replay_weight 10 --vae_weight 1 --edge_weight 1 \
    --lr_model 0.0001 --wd_model 0.0005 --lr_mem 0.001 --wd_mem 0.0005 \
    --warmup_epochs 2 --inner_loop 10

run_case main_gcal.py gcal "" fb100 gcn \
    --test_epochs 10 --train_epochs 150 \
    --entropy_weight 5 --replay_weight 1 --vae_weight 1 --edge_weight 5 \
    --lr_model 0.0001 --wd_model 0.0005 --lr_mem 0.001 --wd_mem 0.0005 \
    --warmup_epochs 9 --inner_loop 1

run_case main_gcal.py gcal "" ogbn-arxiv sage \
    --test_epochs 3 --train_epochs 200 \
    --entropy_weight 5 --replay_weight 10 --vae_weight 1 --edge_weight 1 \
    --lr_model 0.0001 --wd_model 0.0005 --lr_mem 0.001 --wd_mem 0.0005 \
    --warmup_epochs 2 --inner_loop 10

run_case main_gcal.py gcal "" elliptic gcn \
    --test_epochs 5 --train_epochs 200 \
    --entropy_weight 1 --replay_weight 2 --vae_weight 2 --edge_weight 1 \
    --lr_model 0.0001 --wd_model 0.0005 --lr_mem 0.001 --wd_mem 0.0005 \
    --warmup_epochs 3 --inner_loop 50
