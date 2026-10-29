"""Closed-form graph condensation used to build EMC's memory.

The routine performs multi-hop feature propagation, class-wise node
partitioning, redundancy filtering, and a closed-form least-squares solve for
the condensed adjacency and features.  See ``EMC`` in ``methods.py``.
"""

import faiss
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch_geometric.data import Data


def graph_condense(args, data):
    # args.num_class = int(data.y.max()+1)
    # print(f"num_class={args.num_class}")
    args, label_cond = generate_labels_syn(args, data)
    # print(f"label_cond.shape={label_cond.shape}")
    H = conv_graph_multi(data)
    model = linear_model(H, data, args)
    H_aug, y_aug, conf = data_assessment(args, data, model, H)
    # print("start to generate mask...") # slow very slow
    M_norm = mask_generation_conf(H_aug, y_aug, args, 'spectral', conf)
    h = torch.spmm(M_norm.to(args.device), H_aug.to(args.device))

    # # print("start to generate adj...")
    # if args.generate_adj == 1:
    #     a = get_adj(h, args.adj_T)
    #     x = get_feature(a, h, args.alpha)
    #     # graph = Data(x=x, y=label_cond, edge_index=a.nonzero().t(),
    #     #              edge_attr=a[a.nonzero()[:, 0], a.nonzero()[:, 1]],
    #     #              train_mask=torch.ones(len(x), dtype=torch.bool))
    #     graph = Data(x=x, edge_index=a.nonzero().t(), edge_attr=a[a.nonzero()[:, 0], a.nonzero()[:, 1]])
    # else:
    #     # graph = Data(x=h, y=label_cond, edge_index=torch.eye(len(h)).nonzero().t(),
    #     #              edge_attr=torch.ones(len(h)), train_mask=torch.ones(len(h), dtype=torch.bool))
    #     graph = Data(x=h, edge_index=torch.eye(len(h)).nonzero().t(), edge_attr=torch.ones(len(h)))
    # return graph


    ### DRAW
    if args.generate_adj == 1:
        a = get_adj(h, args.adj_T)
        x = get_feature(a, h, args.alpha)
        graph = Data(x=x,y=torch.tensor(label_cond, dtype=torch.long, device=x.device),
            edge_index=a.nonzero().t(),
            edge_attr=a[a.nonzero()[:, 0], a.nonzero()[:, 1]]
        )
    else:
        graph = Data(x=h,y=torch.tensor(label_cond, dtype=torch.long, device=h.device),
        edge_index=torch.eye(len(h), device=h.device).nonzero().t(),
        edge_attr=torch.ones(len(h), device=h.device)
        )
    return graph



def generate_labels_syn(args, data):
    reduction_rate = args.ratio
    from collections import Counter
    # counter = Counter(np.array(data.y[data.train_mask].cpu()))
    counter = Counter(np.array(data.y.cpu()))
    num_class_dict = {}
    # n = len(data.y[data.train_mask])
    n = len(data.y)

    sorted_counter = sorted(counter.items(), key=lambda x:x[1])
    sum_ = 0
    for ix, (c, num) in enumerate(sorted_counter):
        if ix == len(sorted_counter) - 1:
            num_class_dict[c] = int(n * reduction_rate) - sum_
        else:
            num_class_dict[c] = max(int(num * reduction_rate), 1)
            sum_ += num_class_dict[c]

    # print(f"num_class_dict = {num_class_dict}, len = {len(num_class_dict)}")
    # num_class = np.zeros(args.num_class, dtype=int)
    args.num_class = len(num_class_dict)
    num_class = np.zeros(len(num_class_dict), dtype=int)

    for i in range(args.num_class):
        if args.dataset == 'ogbn-arxiv' and args.num_class < 40 and i == 12:
            # Merge the rare class 39 into class 12 when the graph is missing
            # some of the 40 arXiv categories.
            num_class[i] = num_class_dict.get(39, 1)
        elif args.dataset == 'elliptic' and i == 2 and -1 in num_class_dict:
            # Unknown nodes are treated as a separate class during
            # condensation.
            num_class[i] = num_class_dict[-1]
        else:
            num_class[i] = num_class_dict.get(i, 1)

    labels_syn = []
    # for i in range(args.num_class):
    #     labels_syn += [i] * num_class[i]
    # labels_syn = torch.tensor(labels_syn).long()
    # tensor([0, 0, 0,  ..., 1, 1, 1]) torch.Size([1781])

    args.budget = sum(num_class)
    args.budget_cla = num_class
    return args, labels_syn



def conv_graph_multi(data):
    adj_norm = normalize_adj_sparse(data).to(data.x.device)
    H0 = data.x
    H1 = torch.spmm(adj_norm, H0)
    H2 = torch.spmm(adj_norm, H1)
    return H0, H1, H2


def normalize_adj_sparse(data):
    try:
        mx = sp.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index.cpu().numpy()), shape=(data.x.shape[0], data.x.shape[0]))
        if type(mx) is not sp.lil.lil_matrix:
            mx = mx.tolil()
        if mx[0, 0] == 0 :
            mx = mx + sp.eye(mx.shape[0])
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1/2).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        mx = mx.dot(r_mat_inv)

        sparse_mx = mx.tocoo().astype(np.float32)
        sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
        sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
        sparseconcat=torch.cat((sparserow, sparsecol),1)
        sparsedata=torch.FloatTensor(sparse_mx.data)
        adj = torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

    except Exception:
        adj = normalize_sparse_tensor(data.edge_index)
    return adj


def get_domain_y(data, args):
    y = data.y.clone()
    if args.dataset == 'ogbn-arxiv' and args.num_class < 40:
        y[y == 39] = 12

    elif args.dataset == 'elliptic':
        y[y == -1] = 2
    return y


def linear_model(H, data, args):
    feat = sum(H)/len(H)
    # Y_L = torch.nn.functional.one_hot(data.y[ref_nodes], num_class).float()
    # W = torch.linalg.lstsq(feat[ref_nodes.cpu()].cpu(), Y_L.cpu(), driver="gelss")[0]
    domain_y = get_domain_y(data, args)
    Y_L = torch.nn.functional.one_hot(domain_y, args.num_class).float()
    W = torch.linalg.lstsq(feat.cpu(), Y_L.cpu(), driver="gelss")[0]
    return  W



def sample_k_nodes_per_label(label, visible_nodes, k, num_class):
    ref_node_idx = [
        (label[visible_nodes] == lbl).nonzero().view(-1) for lbl in range(num_class)
    ]
    sampled_indices = [
        label_indices[torch.randperm(len(label_indices))[:k]]
        for label_indices in ref_node_idx
    ]
    return visible_nodes[torch.cat(sampled_indices)]


def add_self_loops_(edge_index, edge_weight=None, fill_value=1, num_nodes=None):

    loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


def normalize_sparse_tensor(adj, fill_value=1):
    """Normalize sparse tensor. Need to import torch_scatter
    """
    row, col, _ = adj.coo()
    edge_index = torch.stack([row, col])
    num_nodes= adj.sizes()[0]
    edge_weight = torch.ones(adj.nnz()).to(adj.device())

    edge_index, edge_weight = add_self_loops_(
	edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    from torch_scatter import scatter_add
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    shape = adj.sizes()
    return torch.sparse.FloatTensor(edge_index, values, shape)


def mask_generation_conf(H, y, args, method, conf):
    budgets = args.budget_cla
    # print(f"Budgets={budgets}")
    idx = torch.arange(len(H))
    indices = torch.LongTensor().new_empty((0, 2))
    values = torch.FloatTensor()
    row = 0
    for cls in range(args.num_class):
        cls_mask = (y == cls).cpu()
        H_cls = H[cls_mask].cpu().detach().numpy()
        # cluster_labels = clustering(H_cls, budgets[cls], method)
        cluster_labels = clustering_fast(H_cls, budgets[cls], method)
        for center_idx in range(budgets[cls]):
            center_mask = torch.from_numpy(cluster_labels == center_idx)
            idx_center = idx[cls_mask][center_mask]
            cscore = conf[cls_mask][center_mask]
            if len(idx_center) > 0:
                # val = 1.0 / len(idx_center)
                val = F.softmax(cscore/args.tau, dim=-1)          #calibration
                cls_indices = torch.full((len(idx_center), 1), row, dtype=torch.long)
                combined_indices = torch.cat([cls_indices, idx_center.unsqueeze(1)], dim=1)
                indices = torch.cat([indices, combined_indices], dim=0)
                values = torch.cat([values, val])

            row+=1
    size = torch.Size([args.budget, len(H)])
    mapping_norm = torch.sparse_coo_tensor(indices.t(), values, size)
    return mapping_norm


def data_assessment(args, data, model, feat):

    if args.aug_ratio >0:
        # H_augs = feat[1][data.train_mask]
        # y = data.y[data.train_mask]
        H_augs = feat[1]
        y = get_domain_y(data, args)

        logits_aug = H_augs @ model.to(args.device)
        logits_aug = logits_aug.softmax(dim=-1)
        y_pred = logits_aug.argmax(1)
        report = classification_report(y.cpu().numpy(), y_pred.cpu().numpy(), output_dict=True, zero_division=0)
        acc = [float(metrics['f1-score']) for label, metrics in report.items() if label.isdigit()]
        weights = 1 - np.array(acc)
        probabilities = weights[y.cpu()]+1e-20
        pro = probabilities/sum(probabilities)
        idx = np.random.choice(len(pro), size=int(len(H_augs)*args.aug_ratio), p=pro)

        #process aug
        H_aug = H_augs[idx]
        y_aug = y[idx]
        logits_aug = logits_aug[idx]
        y_pred_aug = logits_aug.argmax(1)
        conf_pred_aug = logits_aug[np.arange(len(logits_aug)), y_aug].cpu()
        conf_pred_aug[y_pred_aug!=y_aug]=torch.min(conf_pred_aug)

        # H = feat[2][data.train_mask]
        H = feat[2]
        logits = H @ model.to(args.device)
        logits = logits.softmax(dim=-1)
        y_pred = logits.argmax(1)
        conf_pred = logits[np.arange(len(logits)), y].cpu()
        conf_pred[y_pred!=y]=torch.min(conf_pred)

        H_all = torch.concat([H, H_aug])
        y_all = torch.concat([y, y_aug])
        conf_all = torch.concat([conf_pred, conf_pred_aug])

    else:
        # y = data.y[data.train_mask]
        # H = feat[2][data.train_mask]
        y = get_domain_y(data, args)
        H = feat[2]
        logits = H @ model.to(args.device)
        logits = logits.softmax(dim=-1)
        y_pred = logits.argmax(1)
        conf_pred = logits[np.arange(len(logits)), y].cpu()
        conf_pred[y_pred!=y]=torch.min(conf_pred)

        H_all = H
        y_all = y
        conf_all = conf_pred

    return H_all, y_all, conf_all


def square_feat_map(z, c=2**-.5):
  polf = PolynomialFeatures(include_bias=True)
  x = polf.fit_transform(z)
  coefs = np.ones(len(polf.powers_))
  coefs[0] = c
  coefs[(polf.powers_ == 1).sum(1) == 2] = np.sqrt(2)
  coefs[(polf.powers_ == 1).sum(1) == 1] = np.sqrt(2*c)
  return x * coefs


def clustering_fast(H, num_center, method):

    if num_center == len(H):
        return np.arange(num_center)

    if method == 'kmeans':
        kmeans = faiss.Kmeans(int(H.shape[1]), int(num_center), gpu=False)
        kmeans.cp.min_points_per_centroid = 1
        kmeans.train(H.astype('float32'))
        _, I = kmeans.index.search(H.astype('float32'), 1)
        cluster_labels = I.flatten()
    elif method == 'spectral':
        H = StandardScaler(with_std=False).fit_transform(H)
        svd = TruncatedSVD(num_center)
        svd.fit(H.T)
        U = svd.components_.T

        Z = square_feat_map(U)
        r = Z.sum(0)
        D = Z @ r
        Z_hat = Z / D[:,None]**.5

        svd = TruncatedSVD(num_center+1)
        svd.fit(Z_hat.T)
        Q = svd.components_.T[:,1:]
        kmeans = faiss.Kmeans(int(Q.shape[1]), int(num_center), gpu=False)
        kmeans.cp.min_points_per_centroid = 1
        kmeans.train(Q.astype('float32'))
        _, I = kmeans.index.search(Q.astype('float32'), 1)
        cluster_labels = I.flatten()
    elif method == 'nocluster':
        samples = np.arange(int(H.shape[0]))
        np.random.shuffle(samples)
        cluster_labels = np.zeros(int(H.shape[0]), dtype=int)

        cluster_size = int (int(H.shape[0]) // int(num_center))

        for i in range(int(num_center)):
            cluster_labels[samples[i * cluster_size : (i + 1) * cluster_size]] = i

    return cluster_labels


def get_adj(h, adj_T):
    h = F.normalize(h, dim=1)
    a = torch.mm(h, h.t())
    adj = (a>adj_T).float()
    adj = adj - torch.diag(torch.diag(adj, 0))
    return adj


def get_feature(a, h, alpha):
    lap = laplacian_tensor(a)
    a_norm = normalize_adj_tensor(a)
    a_conv = a_norm
    for _ in range(2):
        a_conv = torch.mm(a_conv, a_norm)
    aTa = torch.mm(a_conv, a_conv.t())
    aTh = alpha*torch.mm(a_conv.t(), h)
    k = (aTa+lap)
    x = torch.linalg.solve(k, aTh)
    return x


def laplacian_tensor(adj):
    r_inv = adj.sum(1).flatten()
    D = torch.diag(r_inv)
    L = D - adj

    r_inv = r_inv  + 1e-10
    r_inv = r_inv.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    L_norm = r_mat_inv @ L @ r_mat_inv
    return L_norm


def normalize_adj_tensor(adj):
    device = adj.device

    mx = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx

