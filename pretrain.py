"""Pre-train the source GNN used by EMC and GCAL.

A grid search over hidden size, depth, dropout, learning rate and weight decay
is run on the source domains; the configuration with the best validation score
is stored in ``checkpoints/{dataset}_{model}_best.pt``.  The checkpoint is
reused automatically by ``main.py``, ``main_gcal.py`` and ``sequence.py``.

Examples
--------
    python pretrain.py --dataset twitch --model gcn --train_epochs 600
    python pretrain.py --dataset fb100 --model gat --gat_heads 2 --hidden_channels 128
"""

import argparse
import os
import random

import numpy as np
import torch
from torch_geometric.data import Batch

from data_process import load_data
from model import Model
from validation import validation

GRID = {
    'hidden_channels': [64, 128, 256, 512],
    'num_layers': [2, 3, 4, 5],
    'dropout': [0.2, 0.5],
    'lr': [0.0005, 0.001, 0.005],
    'weight_decay': [0.0001, 0.0005],
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_node_split(merged_graph, dataset):
    if dataset == 'elliptic':
        indices = torch.where(merged_graph.mask)[0]
        indices = indices[torch.randperm(len(indices))]
    else:
        indices = torch.randperm(merged_graph.num_nodes)

    train_end = int(0.6 * len(indices))
    val_end = int(0.8 * len(indices))
    train_mask, val_mask, test_mask = (
        torch.zeros(merged_graph.num_nodes, dtype=torch.bool) for _ in range(3)
    )
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True
    merged_graph.train_mask = train_mask
    merged_graph.val_mask = val_mask
    merged_graph.test_mask = test_mask
    return merged_graph


def train_and_validate(args, merged_graph, feature_dim, num_labels, config):
    for key, value in config.items():
        setattr(args, key, value)

    model = Model(args, feature_dim, num_labels).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric, test_metric, best_state = -1.0, None, None
    for epoch in range(args.train_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(merged_graph.x, merged_graph.edge_index, merged_graph.y, merged_graph.train_mask)
        loss.backward()
        optimizer.step()

        if epoch % args.show_interval == 0:
            model.eval()
            pred = model.inference(merged_graph.x, merged_graph.edge_index).argmax(dim=1)
            metric = validation(args, pred, merged_graph.y, merged_graph, merged_graph.val_mask)
            if metric > best_metric:
                best_metric = metric
                test_metric = validation(
                    args, pred, merged_graph.y, merged_graph, merged_graph.test_mask
                )
                best_state = model.state_dict()

    return best_metric, test_metric, best_state


def build_parser():
    parser = argparse.ArgumentParser(description='Source-model pretraining')
    parser.add_argument('--dataset', type=str, default='twitch',
                        choices=['twitch', 'fb100', 'ogbn-arxiv', 'elliptic'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sage', 'gat', 'gin'])
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--show_interval', type=int, default=20)
    parser.add_argument('--model_save_url', type=str, default='checkpoints')

    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--no_bn', action='store_true')
    parser.add_argument('--gat_heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    return parser


def main():
    args = build_parser().parse_args()
    args.device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(args.seed)

    datalist, _ = load_data(args)
    if args.dataset == 'twitch':
        train_dataset = datalist[:1]
    elif args.dataset == 'fb100':
        train_dataset = datalist[:3]
    elif args.dataset == 'ogbn-arxiv':
        train_dataset = datalist[:1]
    else:
        train_dataset = datalist[6:9]

    merged_graph = Batch.from_data_list(train_dataset).to(args.device)
    merged_graph = make_node_split(merged_graph, args.dataset)
    feature_dim = merged_graph.x.shape[1]
    num_labels = int(torch.max(merged_graph.y)) + 1

    best_result = {'metric': -1.0}
    for hidden in GRID['hidden_channels']:
        for layers in GRID['num_layers']:
            for dropout in GRID['dropout']:
                for lr in GRID['lr']:
                    for wd in GRID['weight_decay']:
                        config = {
                            'hidden_channels': hidden,
                            'num_layers': layers,
                            'dropout': dropout,
                            'lr': lr,
                            'weight_decay': wd,
                        }
                        print('Trying config:', config)
                        metric, test_metric, state = train_and_validate(
                            args, merged_graph, feature_dim, num_labels, config
                        )
                        if metric > best_result['metric']:
                            best_result = {
                                'metric': metric,
                                'test_metric': test_metric,
                                'config': config,
                                'state_dict': state,
                            }

    os.makedirs(args.model_save_url, exist_ok=True)
    save_path = os.path.join(args.model_save_url, f'{args.dataset}_{args.model}_best.pt')
    torch.save(best_result['state_dict'], save_path)
    print('Best config:', best_result['config'])
    print(
        'Best validation metric: {:.4f}, test metric: {:.4f}'.format(
            best_result['metric'], best_result['test_metric']
        )
    )
    print('Saved checkpoint to', save_path)


if __name__ == '__main__':
    main()
