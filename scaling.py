"""Scaling study: runtime and peak GPU memory of EMC vs. GCAL.

Synthetic Erdos-Renyi graphs are generated with a controllable number of nodes
and domains.  One graph is used to train a source GCN; the remaining graphs
form a non-stationary sequence.  The script reports the wall-clock time and the
peak GPU memory required to adapt to the whole sequence, which corresponds to
the efficiency results in the paper.

Examples
--------
Node scaling (11 domains):
    python scaling.py --method emc --num_nodes 10000 --num_graphs 11

Domain scaling (10k nodes):
    python scaling.py --method emc --num_nodes 10000 --num_graphs 81
"""

import argparse
import random
import time

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Batch, DataLoader, Dataset
from torch_geometric.utils import from_networkx

from methods import EMC, GCAL
from model import Model


class CustomDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def generate_er_graphs(num_nodes, num_graphs, p=None, num_classes=7, feature_dim=128):
    """Generate ER graphs with balanced class labels."""
    if p is None:
        p = 4.0 / num_nodes

    data_list = []
    for _ in range(num_graphs):
        nx_graph = nx.fast_gnp_random_graph(num_nodes, p)
        data = from_networkx(nx_graph)
        data.x = torch.randn(num_nodes, feature_dim)

        nodes_per_class = num_nodes // num_classes
        y = torch.zeros(num_nodes, dtype=torch.long)
        for cls in range(num_classes):
            start = cls * nodes_per_class
            end = start + nodes_per_class if cls < num_classes - 1 else num_nodes
            y[start:end] = cls
        data.y = y[torch.randperm(num_nodes)]
        data.mask = torch.ones(num_nodes, dtype=torch.bool)
        data_list.append(data)

    return data_list


def quick_train_model(model, graph, epochs=50, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(graph.x, graph.edge_index, graph.y, graph.train_mask)
        loss.backward()
        optimizer.step()
    return model


def build_parser():
    parser = argparse.ArgumentParser(description='EMC scaling efficiency test')
    parser.add_argument('--method', type=str, default='emc', choices=['emc', 'gcal'])

    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sage', 'gat', 'gin'])
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--no_bn', action='store_true')

    # Adaptation hyper-parameters (shared with the main experiments).
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
    parser.add_argument('--syn_ratio', type=float, default=0.15)

    # Synthetic sequence.
    parser.add_argument('--quick_train_epochs', type=int, default=50)
    parser.add_argument('--num_nodes', type=int, default=1000)
    parser.add_argument('--num_graphs', type=int, default=5)
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--edge_prob', type=float, default=None)

    # Compatibility arguments consumed by the condensation routine.
    parser.add_argument('--dataset', type=str, default='er')
    parser.add_argument('--generate_adj', type=int, default=1)
    parser.add_argument('--adj_T', type=float, default=0.95)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--tau', type=float, default=10.0)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--aug_ratio', type=float, default=0.55)
    parser.add_argument('--gat_heads', type=int, default=8)
    parser.add_argument('--model_save_url', type=str, default='checkpoints')
    return parser


def main():
    args = build_parser().parse_args()
    print('=== Scaling efficiency test ===')
    print(f'graph: {args.num_nodes} nodes, {args.num_graphs} graphs')
    print(f'method: {args.method}, model: {args.model}')

    args.device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    datalist = generate_er_graphs(
        num_nodes=args.num_nodes,
        num_graphs=args.num_graphs,
        p=args.edge_prob,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
    )
    train_dataset, test_dataset = datalist[:1], datalist[1:]
    print(f'training graphs: 1, test graphs: {len(test_dataset)}')

    merged_graph = Batch.from_data_list(train_dataset).to(args.device)
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

    model = Model(args, args.feature_dim, args.num_classes).to(args.device)
    print('Training source model...')
    model = quick_train_model(model, merged_graph, epochs=args.quick_train_epochs)

    adaptor_cls = EMC if args.method == 'emc' else GCAL
    cta_model = adaptor_cls(args, model).to(args.device)

    loader = DataLoader(CustomDataset(test_dataset), batch_size=1, shuffle=False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    for i, graph in enumerate(loader):
        print(f'Processing graph {i + 1}/{len(test_dataset)}...')
        cta_model.adapt(graph.to(args.device), epochs=args.test_epochs)

    elapsed = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

    print('\n' + '=' * 50)
    print(f'Method: {args.method.upper()}')
    print(f'Time taken: {elapsed:.2f} seconds')
    print(f'Max GPU memory: {peak_memory:.2f} MiB')
    print(f'Graph size: {args.num_nodes} nodes, {args.num_graphs} graphs, model: {args.model}')
    print('=' * 50)


if __name__ == '__main__':
    main()
