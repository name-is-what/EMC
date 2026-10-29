"""Domain-order and cyclic-shift experiments.

The target domains are presented in a configurable order, optionally repeated
to emulate a long non-stationary stream.  This script is used for the order
sensitivity and cyclic-shift analyses in the paper.

Examples
--------
    python sequence.py --dataset twitch --model gcn --method emc
    python sequence.py --dataset twitch --model gcn --method emc \
        --shuffle_targets --seq_seed 0 --sequence_name twitch_shuffle_0 --save_matrix
    python sequence.py --dataset twitch --model gcn --method emc \
        --repeat_targets 2 --sequence_name twitch_cycle --save_matrix
"""

import argparse
import os
import random
import time
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from data_process import load_twitch
from methods import EMC, GCAL
from model import Model
from validation import validation

warnings.filterwarnings('ignore')


TWITCH_DOMAINS = ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW']


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_twitch_all_domains(data_dir: str):
    return {domain: load_twitch(data_dir, domain) for domain in TWITCH_DOMAINS}


def build_node_split(graph: Data, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(graph.num_nodes, generator=g)

    train_index = indices[: int(0.6 * len(indices))]
    val_index = indices[int(0.6 * len(indices)): int(0.8 * len(indices))]
    test_index = indices[int(0.8 * len(indices)):]

    train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)

    train_mask[train_index] = True
    val_mask[val_index] = True
    test_mask[test_index] = True

    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask
    return graph


# ---------------------------
# Sequence utilities
# ---------------------------
def parse_domain_list(raw: str):
    if raw is None or raw.strip() == '':
        return []
    return [item.strip() for item in raw.split(',') if item.strip()]


def validate_target_domains(domains, source_domain):
    for domain in domains:
        if domain not in TWITCH_DOMAINS:
            raise ValueError(f'Unknown domain in target sequence: {domain}')
        if domain == source_domain:
            raise ValueError(
                f'Source domain {source_domain} should not appear in target sequence.'
            )


def build_target_sequence(args):
    if args.target_seq.strip():
        base_domains = parse_domain_list(args.target_seq)
    else:
        base_domains = [d for d in TWITCH_DOMAINS if d != args.source_domain]

    validate_target_domains(base_domains, args.source_domain)

    if args.shuffle_targets:
        rng = random.Random(args.seq_seed)
        base_domains = base_domains.copy()
        rng.shuffle(base_domains)

    full_sequence = base_domains * args.repeat_targets
    if len(full_sequence) == 0:
        raise ValueError('Target sequence is empty.')
    return base_domains, full_sequence


def make_step_labels(sequence_domains):
    counts = defaultdict(int)
    step_labels = []
    for domain in sequence_domains:
        counts[domain] += 1
        step_labels.append(f'{domain}#{counts[domain]}')
    return step_labels


# ---------------------------
# Training / evaluation
# ---------------------------
def metric_name(dataset: str) -> str:
    if dataset == 'twitch':
        return 'auc'
    if dataset == 'elliptic':
        return 'f1'
    return 'acc'


def safe_cuda_peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def evaluate_graph(args, model, graph: Data, mask=None) -> float:
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph.x, graph.edge_index).argmax(dim=1)
        return float(validation(args, pred, graph.y, graph, mask))


def pretrain_or_load_source_model(args, source_graph_cpu: Data, feature_dim: int, num_labels: int):
    model = Model(args, feature_dim, num_labels).to(args.device)

    ckpt_name = f"{args.dataset}_{args.source_domain}_{args.model}_best.pt"
    model_path = os.path.join(args.model_save_url, ckpt_name)

    source_graph = Batch.from_data_list([deepcopy(source_graph_cpu)]).to(args.device)
    build_node_split(source_graph, args.seed)

    print('----------------- source pre-training ---------------------')
    print(f'source domain: {args.source_domain}')
    print(f'checkpoint: {model_path}')

    if os.path.exists(model_path) and not args.retrain:
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        test_metric = evaluate_graph(args, model, source_graph, source_graph.test_mask)
        print('Loaded existing checkpoint.')
        print(f'Pre-training best {metric_name(args.dataset)}: {test_metric:.4f}')
        return model, source_graph, test_metric, model_path

    os.makedirs(args.model_save_url, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_metric = -1.0
    best_epoch = -1
    best_test_metric = -1.0

    for epoch in tqdm(range(args.train_epochs), ncols=100):
        model.train()
        optimizer.zero_grad()
        loss = model(source_graph.x, source_graph.edge_index, source_graph.y, source_graph.train_mask)
        loss.backward()
        optimizer.step()

        if epoch % args.show_interval == 0:
            val_metric = evaluate_graph(args, model, source_graph, source_graph.val_mask)
            tqdm.write(
                f"Epoch: {epoch:03d}, Loss: {loss.item():.4f}, {metric_name(args.dataset)}: {val_metric:.4f}"
            )
            if val_metric > best_metric:
                best_metric = val_metric
                best_epoch = epoch
                best_test_metric = evaluate_graph(args, model, source_graph, source_graph.test_mask)
                torch.save(model.state_dict(), model_path)

    if best_epoch < 0:
        best_test_metric = evaluate_graph(args, model, source_graph, source_graph.test_mask)
        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path, map_location=args.device))

    print(
        f'Pre-training best epoch: {best_epoch:03d}, '
        f'best_{metric_name(args.dataset)}: {best_test_metric:.4f}'
    )
    return model, source_graph, best_test_metric, model_path


def maybe_reset_peak_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def compute_summary_metrics(results_matrix_np: np.ndarray):
    t = results_matrix_np.shape[0] - 1

    mean_gain = 0.0
    mean_results = results_matrix_np[0, 0]
    for i in range(results_matrix_np.shape[0]):
        if i == 0:
            mean_gain = 0.0
        else:
            mean_gain = -np.mean(np.diag(results_matrix_np)[:i] - results_matrix_np[i][:i])
        mean_results = np.mean(results_matrix_np[i, :i + 1])

    tda_values = [results_matrix_np[i, i] for i in range(1, t + 1)]
    mean_tda = float(np.mean(tda_values)) if len(tda_values) > 0 else 0.0

    tdg_values = []
    for i in range(1, t + 1):
        for j in range(0, i):
            if i != j:
                tdg_values.append(results_matrix_np[i, j])
    mean_tdg = float(np.mean(tdg_values)) if len(tdg_values) > 0 else 0.0

    fa_values = []
    for i in range(1, t):
        for j in range(i + 1, t + 1):
            fa_values.append(results_matrix_np[j, i])
    mean_fa = float(np.mean(fa_values)) if len(fa_values) > 0 else 0.0

    return {
        'final_acc': float(mean_results),
        'final_gain': float(mean_gain),
        'final_TDA': mean_tda,
        'final_TDG': mean_tdg,
        'final_FA': mean_fa,
    }


def save_trial_outputs(save_dir, sequence_name, trial_idx, step_labels, results_matrix):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'{sequence_name}_trial{trial_idx + 1}_matrix.csv')

    header = ['row/col', 'source'] + step_labels
    with open(out_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for i in range(results_matrix.shape[0]):
            row_name = 'source' if i == 0 else step_labels[i - 1]
            row_vals = [row_name] + [f'{x:.6f}' for x in results_matrix[i]]
            f.write(','.join(row_vals) + '\n')


def run_single_trial(
    args,
    base_state_dict,
    source_graph,
    source_metric,
    feature_dim,
    num_labels,
    sequence_graphs,
    step_labels,
):
    fresh_model = Model(args, feature_dim, num_labels).to(args.device)
    fresh_model.load_state_dict(deepcopy(base_state_dict))

    adaptor_cls = EMC if args.method == 'emc' else GCAL
    cta_model = adaptor_cls(args, fresh_model).to(args.device)

    t = len(sequence_graphs)
    results_matrix = np.zeros((1 + t, 1 + t), dtype=np.float64)
    results_matrix[0, 0] = source_metric

    maybe_reset_peak_memory_stats()
    start_time = time.time()

    for i, target_graph in enumerate(sequence_graphs):
        target_graph = deepcopy(target_graph).to(args.device)
        cta_model.adapt(target_graph, epochs=args.test_epochs)

        source_metric = evaluate_graph(args, cta_model.base_model, source_graph, source_graph.val_mask)
        results_matrix[i + 1, 0] = source_metric

        for j in range(i + 1):
            past_graph = deepcopy(sequence_graphs[j]).to(args.device)
            results_matrix[i + 1, j + 1] = evaluate_graph(args, cta_model.base_model, past_graph)

        mean_results = np.mean(results_matrix[i + 1, : i + 2])
        print(f'{step_labels[i]:<12} mean_{metric_name(args.dataset)}: {mean_results:.4f}')

    elapsed = time.time() - start_time
    max_gpu_memory = safe_cuda_peak_memory_mb()
    summary = compute_summary_metrics(results_matrix)
    summary['time_seconds'] = float(elapsed)
    summary['max_gpu_memory_mb'] = float(max_gpu_memory)
    return results_matrix, summary


def aggregate_trial_summaries(summaries):
    result = {}
    for key in ['final_acc', 'final_gain', 'final_TDA', 'final_TDG', 'final_FA']:
        values = np.array([s[key] for s in summaries], dtype=np.float64)
        result[key + '_mean'] = float(values.mean())
        result[key + '_std'] = float(values.std())
        result[key + '_max'] = float(values.max())
    times = np.array([s['time_seconds'] for s in summaries], dtype=np.float64)
    mems = np.array([s['max_gpu_memory_mb'] for s in summaries], dtype=np.float64)
    result['time_seconds_mean'] = float(times.mean())
    result['max_gpu_memory_mb_mean'] = float(mems.mean())
    return result


def build_parser():
    parser = argparse.ArgumentParser(description='Domain-order and cyclic-shift sequence experiment (Twitch)')

    # basic
    parser.add_argument('--dataset', type=str, default='twitch')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_trial', type=int, default=1)

    # pretrain
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--show_interval', type=int, default=20)
    parser.add_argument('--retrain', action='store_true', default=False,
                        help='ignore a cached source checkpoint')
    parser.add_argument('--woTest', type=int, default=0)
    parser.add_argument('--model_save_url', type=str, default='checkpoints')

    # model
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--no_bn', action='store_true')
    parser.add_argument('--gat_heads', type=int, default=8)

    # adaptation
    parser.add_argument('--method', type=str, default='emc', choices=['emc', 'gcal'])
    parser.add_argument('--test_epochs', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--inner_loop', type=int, default=10)
    parser.add_argument('--entropy_weight', type=float, default=5)
    parser.add_argument('--replay_weight', type=float, default=10)
    parser.add_argument('--vae_weight', type=float, default=1)
    parser.add_argument('--edge_weight', type=float, default=1)
    parser.add_argument('--lr_model', type=float, default=0.0001)
    parser.add_argument('--wd_model', type=float, default=0.0005)
    parser.add_argument('--lr_mem', type=float, default=0.001)
    parser.add_argument('--wd_mem', type=float, default=0.0005)

    # graph condensation compatibility
    parser.add_argument('--generate_adj', type=int, default=1)
    parser.add_argument('--adj_T', type=float, default=0.95)
    parser.add_argument('--alpha', type=float, default=3)
    parser.add_argument('--tau', type=float, default=10)
    parser.add_argument('--ratio', type=float, default=0.01)
    parser.add_argument('--aug_ratio', type=float, default=0.55)

    # sequence
    parser.add_argument('--source_domain', type=str, default='DE')
    parser.add_argument('--target_seq', type=str, default='')
    parser.add_argument('--shuffle_targets', action='store_true')
    parser.add_argument('--repeat_targets', type=int, default=1)
    parser.add_argument('--seq_seed', type=int, default=0)
    parser.add_argument('--sequence_name', type=str, default='twitch_sequence')

    # save
    parser.add_argument('--save_dir', type=str, default='results/sequence')
    parser.add_argument('--save_matrix', action='store_true')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset != 'twitch':
        raise ValueError('This sequence experiment is currently implemented for Twitch only.')
    if args.source_domain not in TWITCH_DOMAINS:
        raise ValueError(f'Unknown source domain: {args.source_domain}')

    args.device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    seed_everything(args.seed)

    base_domains, sequence_domains = build_target_sequence(args)
    step_labels = make_step_labels(sequence_domains)

    print(args)
    print('================ sequence setup ================')
    print(f'source_domain: {args.source_domain}')
    print(f'base_targets : {base_domains}')
    print(f'full_sequence: {sequence_domains}')
    print('===============================================')

    domain_graphs = load_twitch_all_domains(args.data_dir)
    source_graph_cpu = domain_graphs[args.source_domain]
    sequence_graphs_cpu = [domain_graphs[d] for d in sequence_domains]

    num_labels = int(source_graph_cpu.y.max().item()) + 1
    feature_dim = int(source_graph_cpu.x.shape[1])
    args.feature_dim = feature_dim

    pretrained_model, source_graph, _, model_path = pretrain_or_load_source_model(
        args=args,
        source_graph_cpu=source_graph_cpu,
        feature_dim=feature_dim,
        num_labels=num_labels,
    )
    base_state_dict = deepcopy(pretrained_model.state_dict())
    source_val_metric = evaluate_graph(args, pretrained_model, source_graph, source_graph.val_mask)

    if args.woTest:
        print('Skipping adaptation/testing as requested by --woTest.')
        return

    print('----------------- sequential testing ---------------------')
    print(f'method: {args.method}')
    print(f'loaded source checkpoint: {model_path}')

    trial_summaries = []
    for trial in range(args.num_trial):
        print(f'================ Trial {trial + 1}/{args.num_trial} ================')
        results_matrix, summary = run_single_trial(
            args=args,
            base_state_dict=base_state_dict,
            source_graph=source_graph,
            source_metric=source_val_metric,
            feature_dim=feature_dim,
            num_labels=num_labels,
            sequence_graphs=sequence_graphs_cpu,
            step_labels=step_labels,
        )
        print('=================')
        print(f"time taken: {summary['time_seconds']:.2f} s, max GPU memory: {summary['max_gpu_memory_mb']:.2f} MiB")
        print('=================')
        print(f"trial_final_acc: {summary['final_acc'] * 100:.2f}")
        print(f"trial_final_gain: {summary['final_gain'] * 100:.2f}")
        print(f"trial_final_TDA: {summary['final_TDA'] * 100:.2f}")
        print(f"trial_final_TDG: {summary['final_TDG'] * 100:.2f}")
        print(f"trial_final_FA: {summary['final_FA'] * 100:.2f}")

        if args.save_matrix:
            save_trial_outputs(args.save_dir, args.sequence_name, trial, step_labels, results_matrix)

        trial_summaries.append(summary)

    final_summary = aggregate_trial_summaries(trial_summaries)
    print('================ Final Results ================')
    print(f"final_acc: {final_summary['final_acc_mean'] * 100:.2f} ± {final_summary['final_acc_std'] * 100:.2f}, max: {final_summary['final_acc_max'] * 100:.2f}")
    print(f"final_gain: {final_summary['final_gain_mean'] * 100:.2f} ± {final_summary['final_gain_std'] * 100:.2f}")
    print(f"final_TDA: {final_summary['final_TDA_mean'] * 100:.2f} ± {final_summary['final_TDA_std'] * 100:.2f}")
    print(f"final_TDG: {final_summary['final_TDG_mean'] * 100:.2f} ± {final_summary['final_TDG_std'] * 100:.2f}")
    print(f"final_FA: {final_summary['final_FA_mean'] * 100:.2f} ± {final_summary['final_FA_std'] * 100:.2f}")
    print(f"avg_time_seconds: {final_summary['time_seconds_mean']:.2f}")
    print(f"avg_max_gpu_memory_mb: {final_summary['max_gpu_memory_mb_mean']:.2f}")
    print('===============================================')


if __name__ == '__main__':
    main()
