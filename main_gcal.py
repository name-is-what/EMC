"""Baseline experiment: GCAL on the same non-stationary graph sequences.

This entry point reproduces the GCAL baseline used for comparison in the
paper.  It shares the source-model training, evaluation and summary utilities
with ``main.py`` and swaps the test-time adapter from EMC to GCAL.
"""

import argparse
import time
import warnings
from copy import deepcopy

import numpy as np
import torch
from torch_geometric.data import Batch, DataLoader

from data_process import load_data
from main import (
    CustomDataset,
    evaluate,
    make_node_split,
    set_seed,
    split_dataset,
    summarize_matrix,
    train_source_model,
)
from methods import GCAL
from model import Model

warnings.filterwarnings('ignore')


def build_parser():
    parser = argparse.ArgumentParser(description='GCAL baseline experiment')
    parser.add_argument('--dataset', type=str, default='twitch',
                        choices=['twitch', 'fb100', 'ogbn-arxiv', 'elliptic'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sage', 'gat', 'gin'])
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_trial', type=int, default=1)

    # Source-model training.
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--no_bn', action='store_true')
    parser.add_argument('--gat_heads', type=int, default=8)
    parser.add_argument('--show_interval', type=int, default=20)
    parser.add_argument('--model_save_url', type=str, default='checkpoints')
    parser.add_argument('--retrain', action='store_true')

    # Test-time adaptation with GCAL.
    parser.add_argument('--test_epochs', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--inner_loop', type=int, default=10)
    parser.add_argument('--entropy_weight', type=float, default=5.0)
    parser.add_argument('--replay_weight', type=float, default=10.0)
    parser.add_argument('--vae_weight', type=float, default=1.0)
    parser.add_argument('--edge_weight', type=float, default=1.0)
    parser.add_argument('--lr_model', type=float, default=0.0001)
    parser.add_argument('--wd_model', type=float, default=0.0005)
    parser.add_argument('--lr_mem', type=float, default=0.001)
    parser.add_argument('--wd_mem', type=float, default=0.0005)

    parser.add_argument('--woTest', action='store_true', help='pretrain only, skip adaptation')
    return parser


def main():
    args = build_parser().parse_args()
    print(args)

    args.device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(args.seed)

    datalist, datanamelist = load_data(args)
    train_dataset, test_dataset, datanamelist = split_dataset(args, datalist, datanamelist)
    print('domains:', datanamelist)

    num_labels = int(torch.max(train_dataset[0].y)) + 1
    feature_dim = train_dataset[0].x.shape[1]
    args.feature_dim = feature_dim

    merged_graph = Batch.from_data_list(train_dataset).to(args.device)
    merged_graph = make_node_split(merged_graph, args)
    model, _ = train_source_model(args, merged_graph, feature_dim, num_labels)
    source_val_metric = evaluate(args, model, merged_graph, merged_graph.val_mask)

    if args.woTest:
        print('Skipping adaptation (--woTest).')
        return

    print('----------------- adaptation (GCAL) ---------------------')
    base_state = deepcopy(model.state_dict())
    test_loader = DataLoader(CustomDataset(test_dataset), batch_size=1, shuffle=False)
    trial_metrics = {key: [] for key in ('acc', 'gain', 'tda', 'tdg', 'fa')}

    for trial in range(args.num_trial):
        print(f'================ Trial {trial + 1}/{args.num_trial} ================')
        trial_model = Model(args, feature_dim, num_labels).to(args.device)
        trial_model.load_state_dict(base_state)
        cta_model = GCAL(args, trial_model).to(args.device)
        results = np.zeros((1 + len(test_dataset), 1 + len(test_dataset)), dtype=np.float64)
        results[0, 0] = source_val_metric

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        for i, graph in enumerate(test_loader):
            target_graph = graph.to(args.device)
            cta_model.adapt(target_graph, epochs=args.test_epochs)

            cta_model.base_model.eval()
            with torch.no_grad():
                results[i + 1, 0] = evaluate(
                    args, cta_model.base_model, merged_graph, merged_graph.val_mask
                )
                for j in range(i + 1):
                    past_graph = test_dataset[j].to(args.device)
                    results[i + 1, j + 1] = evaluate(args, cta_model.base_model, past_graph)

            mean_score = float(np.mean(results[i + 1, :i + 2]))
            print(f'{str(datanamelist[i + 1]):<15} running mean score: {mean_score:.4f}')

        elapsed = time.time() - start_time
        peak_memory = (
            torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        )
        print(f'time: {elapsed:.2f} s, peak GPU memory: {peak_memory:.2f} MiB')

        for key, value in zip(('acc', 'gain', 'tda', 'tdg', 'fa'), summarize_matrix(results)):
            trial_metrics[key].append(value)

    print('================ Final Results ================')
    for key, label in (('acc', 'Acc'), ('gain', 'Gain'), ('tda', 'TDA'),
                       ('tdg', 'TDG'), ('fa', 'FA')):
        values = np.asarray(trial_metrics[key])
        print(f'{label}: {values.mean() * 100:.2f} ± {values.std() * 100:.2f}')
    print('===============================================')


if __name__ == '__main__':
    main()
