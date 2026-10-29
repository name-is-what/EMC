"""Microbenchmark for the closed-form memory-condensation stage.

This script isolates the cost of EMC's condensation routine (label generation,
feature propagation, closed-form solve, masking and adjacency construction) on
a single synthetic ER graph.  It is useful for profiling the memory and time
breakdown reported in the scaling analysis of the paper.

Example
-------
    python scaling_condense.py --num_nodes 10000 --estimate_nodes 100000000
"""

import argparse
import math
import random
import time
import warnings
from types import SimpleNamespace
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

import graph_cond as gc


def warn(*args, **kwargs):
    pass


warnings.warn = warn

try:
    import resource
except ImportError:
    resource = None

try:
    import psutil
except ImportError:
    psutil = None


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StageTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.start = None

    def __enter__(self):
        maybe_sync(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        maybe_sync(self.device)
        self.elapsed = time.perf_counter() - self.start


def get_current_rss_mib() -> float:
    if psutil is None:
        return float("nan")
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_peak_rss_mib() -> float:
    if resource is None:
        return float("nan")
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: KB, macOS: bytes.
    if value > 10**8:
        return value / 1024 / 1024
    return value / 1024


def dense_matrix_memory_bytes(rows: int, cols: int, dtype_bytes: int = 4) -> int:
    return rows * cols * dtype_bytes



def format_bytes_gib(num_bytes: int) -> float:
    return num_bytes / 1024 / 1024 / 1024



def build_args_namespace(parsed_args: argparse.Namespace, device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=parsed_args.dataset,
        device=device,
        generate_adj=parsed_args.generate_adj,
        adj_T=parsed_args.adj_T,
        alpha=parsed_args.alpha,
        tau=parsed_args.tau,
        ratio=parsed_args.ratio,
        aug_ratio=parsed_args.aug_ratio,
        num_class=parsed_args.num_classes,
    )



def generate_single_er_graph(
    num_nodes: int,
    p: Optional[float],
    num_classes: int,
    feature_dim: int,
    seed: int,
) -> Data:
    if p is None:
        p = 4.0 / num_nodes

    nx_graph = nx.fast_gnp_random_graph(num_nodes, p, seed=seed)
    data = from_networkx(nx_graph)

    if not hasattr(data, "edge_index") or data.edge_index.numel() == 0:
        data.edge_index = torch.empty((2, 0), dtype=torch.long)

    data.x = torch.randn(num_nodes, feature_dim)

    nodes_per_class = num_nodes // num_classes
    y = torch.zeros(num_nodes, dtype=torch.long)
    for cls in range(num_classes):
        start_idx = cls * nodes_per_class
        end_idx = start_idx + nodes_per_class if cls < num_classes - 1 else num_nodes
        y[start_idx:end_idx] = cls
    y = y[torch.randperm(num_nodes)]

    data.y = y
    data.mask = torch.ones(num_nodes, dtype=torch.bool)
    return data



def benchmark_graph_condense(args: SimpleNamespace, data: Data, estimate_nodes: Optional[int] = None) -> Tuple[Data, Dict[str, Union[float, int, str]]]:
    metrics = {}  # type: Dict[str, Union[float, int, str]]

    if args.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.reset_accumulated_memory_stats(args.device)

    rss_before_mib = get_current_rss_mib()
    peak_rss_before_mib = get_peak_rss_mib()

    with StageTimer(args.device) as total_timer:
        with StageTimer(args.device) as timer:
            args, label_cond = gc.generate_labels_syn(args, data)
        metrics["t_generate_labels_syn_sec"] = timer.elapsed

        with StageTimer(args.device) as timer:
            H = gc.conv_graph_multi(data)
        metrics["t_conv_graph_multi_sec"] = timer.elapsed

        feat = sum(H) / len(H)
        metrics["linear_feat_rows"] = int(feat.shape[0])
        metrics["linear_feat_cols"] = int(feat.shape[1])
        metrics["linear_num_classes"] = int(args.num_class)

        with StageTimer(args.device) as timer:
            model = gc.linear_model(H, data, args)
        metrics["t_linear_model_lstsq_sec"] = timer.elapsed

        with StageTimer(args.device) as timer:
            H_aug, y_aug, conf = gc.data_assessment(args, data, model, H)
        metrics["t_data_assessment_sec"] = timer.elapsed

        with StageTimer(args.device) as timer:
            M_norm = gc.mask_generation_conf(H_aug, y_aug, args, "spectral", conf)
            h = torch.spmm(M_norm.to(args.device), H_aug.to(args.device))
        metrics["t_mask_and_aggregate_sec"] = timer.elapsed

        if args.generate_adj == 1:
            with StageTimer(args.device) as timer:
                a = gc.get_adj(h, args.adj_T)
                x = gc.get_feature(a, h, args.alpha)
                condensed_graph = Data(
                    x=x,
                    edge_index=a.nonzero().t(),
                    edge_attr=a[a.nonzero()[:, 0], a.nonzero()[:, 1]],
                )
            metrics["t_build_condensed_adj_sec"] = timer.elapsed
        else:
            with StageTimer(args.device) as timer:
                condensed_graph = Data(
                    x=h,
                    edge_index=torch.eye(len(h), device=h.device).nonzero().t(),
                    edge_attr=torch.ones(len(h), device=h.device),
                )
            metrics["t_build_condensed_adj_sec"] = timer.elapsed

    metrics["t_total_graph_condense_sec"] = total_timer.elapsed

    if args.device.type == "cuda":
        metrics["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024
        metrics["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved(args.device) / 1024 / 1024
    else:
        metrics["peak_gpu_allocated_mib"] = 0.0
        metrics["peak_gpu_reserved_mib"] = 0.0

    metrics["rss_before_mib"] = rss_before_mib
    metrics["rss_after_mib"] = get_current_rss_mib()
    metrics["peak_process_rss_mib"] = get_peak_rss_mib()
    metrics["peak_process_rss_delta_mib"] = metrics["peak_process_rss_mib"] - peak_rss_before_mib

    feat_mem = dense_matrix_memory_bytes(metrics["linear_feat_rows"], metrics["linear_feat_cols"], 4)
    label_mem = dense_matrix_memory_bytes(metrics["linear_feat_rows"], metrics["linear_num_classes"], 4)
    metrics["dense_feat_memory_gib"] = format_bytes_gib(feat_mem)
    metrics["dense_label_memory_gib"] = format_bytes_gib(label_mem)
    metrics["dense_feat_plus_label_memory_gib"] = format_bytes_gib(feat_mem + label_mem)

    if estimate_nodes is not None and estimate_nodes > 0:
        est_feat_mem = dense_matrix_memory_bytes(estimate_nodes, metrics["linear_feat_cols"], 4)
        est_label_mem = dense_matrix_memory_bytes(estimate_nodes, metrics["linear_num_classes"], 4)
        metrics["estimate_nodes"] = int(estimate_nodes)
        metrics["estimate_dense_feat_memory_gib"] = format_bytes_gib(est_feat_mem)
        metrics["estimate_dense_label_memory_gib"] = format_bytes_gib(est_label_mem)
        metrics["estimate_dense_feat_plus_label_memory_gib"] = format_bytes_gib(est_feat_mem + est_label_mem)

    metrics["condensed_num_nodes"] = int(condensed_graph.num_nodes)
    metrics["condensed_num_edges"] = int(condensed_graph.edge_index.shape[1])
    return condensed_graph, metrics



def print_summary(args: argparse.Namespace, graph: Data, metrics: Dict[str, Union[float, int, str]]) -> None:
    print("\n" + "=" * 72)
    print("Graph condensation scaling benchmark")
    print("=" * 72)
    print(f"device: {args.device}")
    print(f"seed: {args.seed}")
    print(f"num_nodes: {graph.num_nodes}")
    print(f"num_edges: {graph.edge_index.shape[1]}")
    print(f"feature_dim: {graph.x.shape[1]}")
    print(f"num_classes: {int(graph.y.max().item()) + 1}")
    print(f"edge_prob: {args.edge_prob if args.edge_prob is not None else 4.0 / args.num_nodes:.8f}")
    print(f"ratio: {args.ratio}")
    print(f"aug_ratio: {args.aug_ratio}")
    print(f"generate_adj: {args.generate_adj}")
    print("-" * 72)
    print(f"time.generate_labels_syn: {metrics['t_generate_labels_syn_sec']:.6f} s")
    print(f"time.conv_graph_multi:   {metrics['t_conv_graph_multi_sec']:.6f} s")
    print(f"time.linear_model:      {metrics['t_linear_model_lstsq_sec']:.6f} s")
    print(f"time.data_assessment:   {metrics['t_data_assessment_sec']:.6f} s")
    print(f"time.mask+aggregate:    {metrics['t_mask_and_aggregate_sec']:.6f} s")
    print(f"time.build_cond_graph:  {metrics['t_build_condensed_adj_sec']:.6f} s")
    print(f"time.total:             {metrics['t_total_graph_condense_sec']:.6f} s")
    print("-" * 72)
    print(f"peak_gpu_allocated:     {metrics['peak_gpu_allocated_mib']:.2f} MiB")
    print(f"peak_gpu_reserved:      {metrics['peak_gpu_reserved_mib']:.2f} MiB")
    if not math.isnan(metrics["rss_before_mib"]):
        print(f"rss_before:             {metrics['rss_before_mib']:.2f} MiB")
    if not math.isnan(metrics["rss_after_mib"]):
        print(f"rss_after:              {metrics['rss_after_mib']:.2f} MiB")
    if not math.isnan(metrics["peak_process_rss_mib"]):
        print(f"peak_process_rss:       {metrics['peak_process_rss_mib']:.2f} MiB")
    if not math.isnan(metrics["peak_process_rss_delta_mib"]):
        print(f"peak_rss_delta:         {metrics['peak_process_rss_delta_mib']:.2f} MiB")
    print("-" * 72)
    print(
        "linear solve shape:      "
        f"A=({metrics['linear_feat_rows']}, {metrics['linear_feat_cols']}), "
        f"B=({metrics['linear_feat_rows']}, {metrics['linear_num_classes']})"
    )
    print(f"dense feat memory:      {metrics['dense_feat_memory_gib']:.4f} GiB")
    print(f"dense label memory:     {metrics['dense_label_memory_gib']:.4f} GiB")
    print(f"dense feat+label:       {metrics['dense_feat_plus_label_memory_gib']:.4f} GiB")
    if "estimate_nodes" in metrics:
        print("-" * 72)
        print(f"estimate_nodes:         {metrics['estimate_nodes']}")
        print(f"est feat memory:        {metrics['estimate_dense_feat_memory_gib']:.2f} GiB")
        print(f"est label memory:       {metrics['estimate_dense_label_memory_gib']:.2f} GiB")
        print(f"est feat+label memory:  {metrics['estimate_dense_feat_plus_label_memory_gib']:.2f} GiB")
    print("-" * 72)
    print(f"condensed_num_nodes:    {metrics['condensed_num_nodes']}")
    print(f"condensed_num_edges:    {metrics['condensed_num_edges']}")
    print("=" * 72)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph condensation scaling benchmark (condensation stage only)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--dataset", type=str, default="er", help="Dataset name for compatibility")

    parser.add_argument("--num_nodes", type=int, default=1000, help="Number of nodes in the single graph")
    parser.add_argument("--feature_dim", type=int, default=128, help="Node feature dimension")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of node classes")
    parser.add_argument("--edge_prob", type=float, default=None, help="ER edge probability; default is 4 / num_nodes")

    parser.add_argument("--generate_adj", type=int, default=1, help="Whether to generate condensed adjacency")
    parser.add_argument("--adj_T", type=float, default=0.95, help="Threshold for condensed adjacency")
    parser.add_argument("--alpha", type=float, default=3.0, help="Feature smoothing weight")
    parser.add_argument("--tau", type=float, default=10.0, help="Calibration temperature")
    parser.add_argument("--ratio", type=float, default=0.1, help="Condensation ratio")
    parser.add_argument("--aug_ratio", type=float, default=0.55, help="Augmentation ratio")

    parser.add_argument(
        "--estimate_nodes",
        type=int,
        default=100000000,
        help="Optional theoretical memory estimate for the linear solve inputs at a larger node count",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    cli_args.device = torch.device(f"cuda:{cli_args.device}") if torch.cuda.is_available() else torch.device("cpu")

    print("=== Graph condensation scaling benchmark ===")
    print(f"single graph only, num_nodes={cli_args.num_nodes}, device={cli_args.device}")

    set_seed(cli_args.seed)

    graph = generate_single_er_graph(
        num_nodes=cli_args.num_nodes,
        p=cli_args.edge_prob,
        num_classes=cli_args.num_classes,
        feature_dim=cli_args.feature_dim,
        seed=cli_args.seed,
    )
    graph = graph.to(cli_args.device)

    args_ns = build_args_namespace(cli_args, cli_args.device)

    try:
        condensed_graph, metrics = benchmark_graph_condense(
            args=args_ns,
            data=graph,
            estimate_nodes=cli_args.estimate_nodes,
        )
        print_summary(cli_args, graph, metrics)
    except RuntimeError as exc:
        print("\n[RuntimeError during benchmark]")
        print(str(exc))
        if cli_args.device.type == "cuda":
            print(f"peak_gpu_allocated_before_error: {torch.cuda.max_memory_allocated(cli_args.device) / 1024 / 1024:.2f} MiB")
            print(f"peak_gpu_reserved_before_error:  {torch.cuda.max_memory_reserved(cli_args.device) / 1024 / 1024:.2f} MiB")
        raise
