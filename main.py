"""Main experiment: EMC on non-stationary graph sequence benchmarks.

The source model is trained on the first domain and then adapted, one domain at
a time, with Efficient Memory Crystallization (EMC).  For every domain the
script reports the accuracy (or AUC / macro-F1) matrix of all previously seen
domains and the summary metrics used in the paper:

    Acc   -- average accuracy over the source domain and all target domains.
    Gain  -- average change of past-domain accuracy after adaptation.
    TDA   -- target-domain accuracy (on-diagonal entries).
    TDG   -- target-domain generalisation (lower triangle).
    FA    -- forgetting on earlier target domains.
"""

import argparse
import os
from copy import deepcopy
import random
import time
import warnings

import numpy as np
import torch
from torch_geometric.data import Batch, DataLoader, Dataset
from tqdm import tqdm

from data_process import load_data
from methods import EMC
from model import Model
from validation import validation

warnings.filterwarnings('ignore')


class CustomDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def build_parser():
    parser = argparse.ArgumentParser(description='EMC main experiment')
    parser.add_argument('--dataset', type=str, default='twitch',
                        choices=['twitch', 'fb100', 'ogbn-arxiv', 'elliptic'])
    parser.add_argument('--data_dir', type=str, default='./data', help='dataset directory')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sage', 'gat', 'gin'])
    parser.add_argument('--device', type=str, default='0', help='CUDA device id')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_trial', type=int, default=5, help='repeated adaptation trials')

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
    parser.add_argument('--retrain', action='store_true', help='ignore a cached source checkpoint')

    # Test-time adaptation with EMC.
    parser.add_argument('--test_epochs', type=int, default=3, help='adaptation epochs per target domain')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='epochs before condensed memories are written')
    parser.add_argument('--entropy_weight', type=float, default=5.0)
    parser.add_argument('--replay_weight', type=float, default=10.0)
    parser.add_argument('--lr_model', type=float, default=0.0001)
    parser.add_argument('--wd_model', type=float, default=0.0005)
    parser.add_argument('--ablation', type=str, default='full',
                        choices=EMC.VALID_ABLATIONS,
                        help='full EMC, or one of its ablations')

    # Closed-form memory condensation.
    parser.add_argument('--generate_adj', type=int, default=1,
                        help='generate a condensed adjacency (0: features only)')
    parser.add_argument('--adj_T', type=float, default=0.95)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--tau', type=float, default=10.0)
    parser.add_argument('--ratio', type=float, default=0.01, help='condensation ratio')
    parser.add_argument('--aug_ratio', type=float, default=0.55)

    parser.add_argument('--woTest', action='store_true', help='pretrain only, skip adaptation')
    return parser


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(args, datalist, datanamelist):
    if args.dataset == 'twitch':
        train_dataset, test_dataset = datalist[:1], datalist[1:]
    elif args.dataset == 'fb100':
        train_dataset, test_dataset = datalist[:3], datalist[3:]
        datanamelist = ['0-2'] + datanamelist[3:]
    elif args.dataset == 'ogbn-arxiv':
        train_dataset, test_dataset = datalist[:1], datalist[1:]
    else:  # elliptic
        train_dataset, test_dataset = datalist[6:9], datalist[9:]
        datanamelist = ['6-8'] + datanamelist[9:]
    return train_dataset, test_dataset, datanamelist


def make_node_split(merged_graph, args):
    if args.dataset == 'elliptic':
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


def evaluate(args, model, graph, mask=None):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph.x, graph.edge_index).argmax(dim=1)
        return validation(args, pred, graph.y, graph, mask)


def train_source_model(args, merged_graph, feature_dim, num_labels):
    model_path = os.path.join(args.model_save_url, f'{args.dataset}_{args.model}_best.pt')
    model = Model(args, feature_dim, num_labels).to(args.device)

    if os.path.exists(model_path) and not args.retrain:
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        test_metric = evaluate(args, model, merged_graph, merged_graph.test_mask)
        print(f'Loaded source checkpoint: {model_path}')
        print(f'Source test score: {test_metric * 100:.2f}')
        return model, test_metric

    os.makedirs(args.model_save_url, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_metric, best_epoch, test_metric = -1.0, -1, None

    for epoch in tqdm(range(args.train_epochs), ncols=100):
        model.train()
        optimizer.zero_grad()
        loss = model(merged_graph.x, merged_graph.edge_index, merged_graph.y, merged_graph.train_mask)
        loss.backward()
        optimizer.step()

        if epoch % args.show_interval == 0:
            metric = evaluate(args, model, merged_graph, merged_graph.val_mask)
            tqdm.write(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, val: {metric:.4f}')
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                test_metric = evaluate(args, model, merged_graph, merged_graph.test_mask)
                torch.save(model.state_dict(), model_path)

    if test_metric is None:
        test_metric = evaluate(args, model, merged_graph, merged_graph.test_mask)
        torch.save(model.state_dict(), model_path)

    print(f'Best epoch: {best_epoch:03d}, source test score: {test_metric * 100:.2f}')
    return model, test_metric


def summarize_matrix(results):
    """Return (acc, gain, TDA, TDG, FA) for a (T+1) x (T+1) result matrix."""
    num_targets = results.shape[0] - 1
    last_row = results[num_targets]
    mean_acc = float(np.mean(last_row))
    mean_gain = float(-np.mean(np.diag(results)[:num_targets] - last_row[:num_targets]))

    tda = [results[i, i] for i in range(1, num_targets + 1)]
    tdg = [results[i, j] for i in range(1, num_targets + 1) for j in range(i)]
    fa = [results[j, i] for i in range(1, num_targets) for j in range(i + 1, num_targets + 1)]

    return (
        mean_acc,
        mean_gain,
        float(np.mean(tda)) if tda else 0.0,
        float(np.mean(tdg)) if tdg else 0.0,
        float(np.mean(fa)) if fa else 0.0,
    )


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
    print(f'num_labels: {num_labels}, feature_dim: {feature_dim}')

    merged_graph = Batch.from_data_list(train_dataset).to(args.device)
    merged_graph = make_node_split(merged_graph, args)
    model, _ = train_source_model(args, merged_graph, feature_dim, num_labels)
    source_val_metric = evaluate(args, model, merged_graph, merged_graph.val_mask)

    if args.woTest:
        print('Skipping adaptation (--woTest).')
        return

    print('----------------- adaptation ---------------------')
    base_state = deepcopy(model.state_dict())
    test_loader = DataLoader(CustomDataset(test_dataset), batch_size=1, shuffle=False)
    trial_metrics = {key: [] for key in ('acc', 'gain', 'tda', 'tdg', 'fa')}

    for trial in range(args.num_trial):
        print(f'================ Trial {trial + 1}/{args.num_trial} ================')
        trial_model = Model(args, feature_dim, num_labels).to(args.device)
        trial_model.load_state_dict(base_state)
        cta_model = EMC(args, trial_model).to(args.device)
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
            name = str(datanamelist[i + 1]).ljust(15)
            tqdm.write(f'{name} running mean score: {mean_score:.4f}')

        elapsed = time.time() - start_time
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_memory = 0.0
        print(f'time: {elapsed:.2f} s, peak GPU memory: {peak_memory:.2f} MiB')

        acc, gain, tda, tdg, fa = summarize_matrix(results)
        for key, value in zip(('acc', 'gain', 'tda', 'tdg', 'fa'), (acc, gain, tda, tdg, fa)):
            trial_metrics[key].append(value)

    print('================ Final Results ================')
    for key, label in (('acc', 'Acc'), ('gain', 'Gain'), ('tda', 'TDA'),
                       ('tdg', 'TDG'), ('fa', 'FA')):
        values = np.asarray(trial_metrics[key])
        print(f'{label}: {values.mean() * 100:.2f} ± {values.std() * 100:.2f}')
    print(f"Acc max: {max(trial_metrics['acc']) * 100:.2f}")
    print('===============================================')


if __name__ == '__main__':
    main()
