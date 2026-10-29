from copy import deepcopy
from itertools import product

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TopKPooling
from torch_geometric.utils import subgraph

from graph_cond import *
from nets import *


class PGE(nn.Module):

    def __init__(self, nfeat, nhid=128, nlayers=3):
        super(PGE, self).__init__()

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat * 2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers - 2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))

        self.reset_parameters()

    def forward(self, x, reparam=True):
        nnodes = x.shape[0]
        edge_index = torch.tensor(list(product(range(nnodes), range(nnodes)))).T.to(x.device)

        edge_embed = torch.cat([x[edge_index[0]], x[edge_index[1]]], axis=1)
        for ix, layer in enumerate(self.layers):
            edge_embed = layer(edge_embed)
            if ix != len(self.layers) - 1:
                edge_embed = self.bns[ix](edge_embed)
                edge_embed = F.relu(edge_embed)
        adj = edge_embed.reshape(nnodes, nnodes)
        adj = (adj + adj.T) / 2
        adj = torch.sigmoid(adj)

        if reparam:
            new_edge_weight = F.gumbel_softmax(adj, 0.5, hard=False)
        else:
            new_edge_weight = F.gumbel_softmax(adj, 0.5, hard=True)

        new_edge_weight = new_edge_weight.reshape(-1)
        mask = new_edge_weight > 0.8
        edge_prob = adj.reshape(-1, 1)
        row, col = edge_index[0], edge_index[1]
        row, col = torch.masked_select(row, mask), torch.masked_select(col, mask)
        new_edge_index = torch.stack([row, col], dim=0)
        new_edge_weight = edge_prob[mask].reshape(-1)
        return new_edge_index, new_edge_weight, edge_prob, mask

    @torch.no_grad()
    def inference(self, x):
        adj_syn = self.forward(x)
        return adj_syn

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)


class GNN_encoder(nn.Module):
    def __init__(self, args, gnn, num_layers, feature_dim, hidden_channels, out_dim):
        super(GNN_encoder, self).__init__()
        self.vgae_gnn = gnn
        if gnn == 'gcn':
            self.gnn = GCN(
                in_channels=feature_dim,
                hidden_channels=hidden_channels,
                out_channels=out_dim,
                num_layers=num_layers,
                dropout=args.dropout,
                use_bn=True,
            )
        elif gnn == 'sage':
            self.gnn = SAGE(
                in_channels=feature_dim,
                hidden_channels=hidden_channels,
                out_channels=out_dim,
                num_layers=num_layers,
                dropout=args.dropout,
            )
        elif gnn == 'gat':
            self.gnn = GAT(
                in_channels=feature_dim,
                hidden_channels=hidden_channels,
                out_channels=out_dim,
                num_layers=num_layers,
                dropout=args.dropout,
                heads=args.gat_heads,
            )
        elif gnn == 'gin':
            self.gnn = GIN(
                in_channels=feature_dim,
                hidden_channels=hidden_channels,
                out_channels=out_dim,
                num_layers=num_layers,
                dropout=args.dropout,
            )
        self.norm = nn.BatchNorm1d(out_dim)

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        if self.vgae_gnn == 'gin':
            out, _ = self.gnn(x, edge_index)
        else:
            out, _ = self.gnn(x, edge_index, edge_weight)
        out = self.norm(out)
        return out


class VGAE(nn.Module):
    def __init__(self, args, nfeat, ratio):
        super(VGAE, self).__init__()
        self.GNN_mu_sigma = GNN_encoder(
            args,
            gnn=args.model,
            num_layers=1,
            feature_dim=nfeat,
            hidden_channels=nfeat,
            out_dim=nfeat * 2,
        )
        self.selector = TopKPooling(in_channels=nfeat * 2, ratio=ratio)
        self.pge = PGE(nfeat=nfeat).to(args.device)
        self.ratio = ratio

    def encoder(self, x, edge_index):
        mu_logvar = self.GNN_mu_sigma(x, edge_index)
        mu_logvar = self.selector(mu_logvar, edge_index)[0]
        mu, logvar = (
            mu_logvar[:, : mu_logvar.shape[1] // 2],
            mu_logvar[:, mu_logvar.shape[1] // 2 :],
        )
        return mu, logvar

    def reparameter(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = eps.mul(std).add_(mu)
        return x

    def decoder(self, mu, logvar):
        new_x = self.reparameter(mu, logvar)
        new_edge_index, new_edge_weight, edge_prob, mask = self.pge(new_x)
        return new_x, new_edge_index, new_edge_weight, edge_prob, mask

    def reparameter_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def kl_divergence_bernoulli(self, edge_prob, q=0.1):
        return torch.sum(edge_prob * torch.log(edge_prob / q) + (1 - edge_prob) * torch.log((1 - edge_prob) / (1 - q)))

    def reset_parameters(self):
        self.GNN_mu_sigma.reset_parameters()
        self.selector.reset_parameters()
        self.pge.reset_parameters()

    def inference(self, mu, std):
        new_edge_index, new_edge_weight, edge_prob, _ = self.pge(mu, False)
        return mu, new_edge_index, new_edge_weight


def MMD_loss(node_embedding, syn_node_embedding):
    mmd = torch.mean((node_embedding.mean(0) - syn_node_embedding.mean(0)) ** 2)
    return mmd


class GCAL(nn.Module):
    def __init__(self, args, base_model):
        super().__init__()
        self.args = args
        self.base_model = base_model

        self.memory = []
        self.random_subgraph_memory = []
        self.feature_avg_memory = []

        self.syn_radio = 0.05

        self.entropy_weight = args.entropy_weight
        self.replay_weight = args.replay_weight
        self.vae_weight = args.vae_weight
        self.edge_weight = args.edge_weight

        self.lr_model = args.lr_model
        self.wd_model = args.wd_model
        self.lr_mem = args.lr_mem
        self.wd_mem = args.wd_mem
        self.warmup_epochs = args.warmup_epochs
        self.inner_loop = args.inner_loop
        self.mt = 0.9

        self.vgae = VGAE(args=self.args, nfeat=args.feature_dim, ratio=self.syn_radio).to(self.args.device)
        if args.dataset == 'twitch' or args.dataset == 'fb100':
            self.base_model = configure_model(self.base_model)
        self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=self.lr_model, weight_decay=self.wd_model)
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = copy_model_and_optimizer(
            self.base_model, self.optimizer
        )

    def adapt(self, target, epochs):
        self.n = max(1, int(target.x.shape[0] * self.syn_radio))
        self.d = target.x.shape[1]
        self.optimizer_m = torch.optim.Adam(self.vgae.parameters(), lr=self.lr_mem, weight_decay=self.wd_mem)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mae_loss = nn.MSELoss()

        for i in range(epochs):
            outputs = self.forward_and_adapt(target, i, epochs)
            # outputs = self.forward_and_adapt_subgraph_sampling(target, i, epochs)
            # outputs = self.forward_and_adapt_feature_averaging(target, i, epochs)

        state_dict = self.model_ema.state_dict()
        self.base_model.load_state_dict(deepcopy(state_dict))
        return outputs

    def _set_bn_eval(self):
        for module in self.base_model.modules():
            if 'BatchNorm' in module._get_name():
                module.eval()

    def _gnn_forward(self, x, edge_index, edge_weight=None):
        if self.args.model == 'gin':
            return self.base_model.gnn(x, edge_index)
        return self.base_model.gnn(x, edge_index, edge_weight)

    def _replay_graph_list(self, graph_list, base_loss):
        loss = base_loss
        for graph in graph_list:
            replay_edge_weight = getattr(graph, 'edge_attr', None)
            outputs, _ = self._gnn_forward(graph.x, graph.edge_index, replay_edge_weight)
            loss = loss + self.replay_weight * self.softmax_entropy(outputs, outputs)
        return loss

    def _make_self_loop_graph(self, x):
        num_nodes = x.size(0)
        if num_nodes == 0:
            x = torch.zeros(1, self.d, device=self.args.device)
            num_nodes = 1
        edge_index = torch.arange(num_nodes, device=x.device, dtype=torch.long).unsqueeze(0).repeat(2, 1)
        edge_attr = torch.ones(edge_index.size(1), device=x.device)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @torch.no_grad()
    def build_random_subgraph_prototype(self, target):
        num_nodes = target.x.size(0)
        k = min(max(1, self.n), num_nodes)
        perm = torch.randperm(num_nodes, device=target.x.device)[:k]
        perm = perm.sort()[0]

        edge_index_sub, edge_attr_sub = subgraph(
            perm,
            target.edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
            return_edge_mask=False,
        )
        x_sub = target.x[perm]

        if edge_index_sub.numel() == 0:
            return self._make_self_loop_graph(x_sub)

        graph = Data(x=x_sub, edge_index=edge_index_sub)
        if edge_attr_sub is not None:
            graph.edge_attr = edge_attr_sub
        return graph

    @torch.no_grad()
    def build_feature_averaging_prototype(self, target):
        self.base_model.eval()
        outputs, _ = self._gnn_forward(target.x, target.edge_index, getattr(target, 'edge_attr', None))
        pseudo_y = outputs.argmax(dim=1)

        proto_x = []
        for c in torch.unique(pseudo_y):
            cls_mask = pseudo_y == c
            if cls_mask.sum() == 0:
                continue
            proto_x.append(target.x[cls_mask].mean(dim=0, keepdim=True))

        if len(proto_x) == 0:
            proto_x = [target.x.mean(dim=0, keepdim=True)]

        proto_x = torch.cat(proto_x, dim=0)
        return self._make_self_loop_graph(proto_x)

    def forward_and_adapt(self, target, i, epochs):
        self.base_model.train()
        outputs, _ = self._gnn_forward(target.x, target.edge_index, getattr(target, 'edge_attr', None))
        self._set_bn_eval()

        loss = self.softmax_entropy(outputs, outputs)

        for (mu, std) in self.memory:
            feat_syn, edge_index_syn, edge_weight_syn = self.vgae.inference(mu, std)
            outputs, _ = self._gnn_forward(feat_syn, edge_index_syn, edge_weight_syn)
            loss = loss + self.replay_weight * self.softmax_entropy(outputs, outputs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.base_model, alpha_teacher=self.mt)

        if i < self.warmup_epochs:
            return outputs

        for il in range(self.inner_loop):
            self.base_model.eval()
            mu, std = self.vgae.encoder(target.x, target.edge_index)
            feat_syn, syn_edge_index, syn_edge_weight, edge_prob, mask = self.vgae.decoder(mu, std)

            outputs, _ = self._gnn_forward(target.x, target.edge_index, getattr(target, 'edge_attr', None))
            loss_real = self.softmax_entropy(outputs, outputs)
            gw_real = torch.autograd.grad(loss_real, self.base_model.parameters())
            gw_real = [_.detach().clone() for _ in gw_real]

            outputs, _ = self._gnn_forward(feat_syn, syn_edge_index, syn_edge_weight)
            loss_syn = self.softmax_entropy(outputs, outputs)
            gw_syn = torch.autograd.grad(loss_syn, self.base_model.parameters(), create_graph=True)
            loss_diff = match_loss(gw_syn, gw_real, 'wb', self.args.device)
            loss_vae = self.vgae.reparameter_loss(mu, std) + self.vgae.kl_divergence_bernoulli(edge_prob)
            _, node_embeddings = self._gnn_forward(target.x, target.edge_index, getattr(target, 'edge_attr', None))
            _, syn_node_embeddings = self._gnn_forward(feat_syn, syn_edge_index, syn_edge_weight)
            loss_edge = MMD_loss(node_embeddings, syn_node_embeddings)
            loss = loss_diff + self.vae_weight * loss_vae + self.edge_weight * loss_edge
            self.optimizer_m.zero_grad()
            loss.backward()
            self.optimizer_m.step()

        if i == epochs - 1:
            mu, std = self.vgae.encoder(target.x, target.edge_index)
            self.memory.append((mu.detach(), std.detach()))
        return outputs

    def forward_and_adapt_subgraph_sampling(self, target, i, epochs):
        self.base_model.train()
        outputs, _ = self._gnn_forward(target.x, target.edge_index, getattr(target, 'edge_attr', None))
        self._set_bn_eval()

        loss = self.softmax_entropy(outputs, outputs)
        loss = self._replay_graph_list(self.random_subgraph_memory, loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.base_model, alpha_teacher=self.mt)

        if i < self.warmup_epochs:
            return outputs

        if i == epochs - 1:
            sampled_graph = self.build_random_subgraph_prototype(target)
            self.random_subgraph_memory.append(sampled_graph)
        return outputs

    def forward_and_adapt_feature_averaging(self, target, i, epochs):
        self.base_model.train()
        outputs, _ = self._gnn_forward(target.x, target.edge_index, getattr(target, 'edge_attr', None))
        self._set_bn_eval()

        loss = self.softmax_entropy(outputs, outputs)
        loss = self._replay_graph_list(self.feature_avg_memory, loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.base_model, alpha_teacher=self.mt)

        if i < self.warmup_epochs:
            return outputs

        if i == epochs - 1:
            avg_graph = self.build_feature_averaging_prototype(target)
            self.feature_avg_memory.append(avg_graph)
        return outputs

    def inference(self, target):
        self.base_model.eval()
        outputs = self.base_model.inference(target.x, target.edge_index)
        return outputs

    def softmax_entropy(self, x, x_ema):
        entropy = -(x_ema.softmax(1) * x.log_softmax(1)).sum(1).mean()
        msfotmax = x.softmax(1).mean(0)
        entropy += self.entropy_weight * (msfotmax * torch.log(msfotmax)).sum()
        return entropy

    def psudo_entropy(self, x):
        y_hat = torch.argmax(x, dim=1)
        entropy = nn.CrossEntropyLoss()(x, y_hat)
        msfotmax = x.softmax(1).mean(0)
        entropy += self.entropy_weight * (msfotmax * torch.log(msfotmax)).sum()
        return entropy


def match_loss(gw_syn, gw_real, dis_metric, device):
    dis = torch.tensor(0.0).to(device)

    if dis_metric == 'wb':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)
    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)
    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
            torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001
        )
    else:
        exit('DC error: unknown distance function')

    return dis


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4:
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:
        tmp = 'do nothing'
    elif len(shape) == 1:
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def normalize_adj_tensor(adj, sparse=False):
    device = adj.device
    if sparse:
        adj = to_scipy(adj)
        mx = normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx


def to_scipy(tensor):
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)


def is_sparse_tensor(tensor):
    return tensor.layout == torch.sparse_coo


def normalize_adj(mx):
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol = torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1)
    sparsedata = torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(), sparsedata, torch.Size(sparse_mx.shape))


def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        for np_name, p in m.named_parameters():
            if np_name in ['weight', 'bias'] and p.requires_grad:
                params.append(p)
                names.append(f"{nm}.{np_name}")
                print(nm, np_name)
    return params, names


class EMC(nn.Module):
    """Efficient Memory Crystallization (EMC).

    During test time the source model is adapted to each incoming target graph
    while a compact memory of previously seen domains is replayed.  The memory
    is produced by closed-form graph condensation, so no auxiliary generative
    module or per-domain re-training is required.

    Args:
        args: experiment arguments (see ``main.py``).
        base_model: source model pre-trained on the first domain.

    ``args.ablation`` selects the variants reported in the ablation study:

    * ``full``            -- EMC with condensed-memory replay (default).
    * ``no_replay``       -- removes replay of the condensed memory.
    * ``no_condensation`` -- replaces condensed memories with raw target
                             graphs, i.e. removes efficient condensation.
    """

    VALID_ABLATIONS = ('full', 'no_replay', 'no_condensation')

    def __init__(self, args, base_model):
        super().__init__()
        self.args = args
        self.base_model = base_model

        self.memory = []
        self.ablation = getattr(args, 'ablation', 'full')
        if self.ablation not in self.VALID_ABLATIONS:
            raise ValueError(
                f'Unknown ablation "{self.ablation}", expected one of {self.VALID_ABLATIONS}'
            )

        self.entropy_weight = args.entropy_weight
        self.replay_weight = args.replay_weight

        self.lr_model = args.lr_model
        self.wd_model = args.wd_model
        self.warmup_epochs = args.warmup_epochs
        self.mt = 0.9

        if args.dataset in ('twitch', 'fb100'):
            self.base_model = configure_model(self.base_model)

        self.optimizer = torch.optim.Adam(
            self.base_model.parameters(), lr=self.lr_model, weight_decay=self.wd_model
        )
        _, _, self.model_ema, _ = copy_model_and_optimizer(self.base_model, self.optimizer)

    def adapt(self, target, epochs):
        if self.ablation == 'full':
            adapt = self.forward_full
        elif self.ablation == 'no_replay':
            adapt = self.forward_no_replay
        else:
            adapt = self.forward_no_condensation

        outputs = None
        for i in range(epochs):
            outputs = adapt(target, i, epochs)

        self.base_model.load_state_dict(deepcopy(self.model_ema.state_dict()))
        return outputs

    def _set_bn_eval(self):
        for module in self.base_model.modules():
            if 'BatchNorm' in module._get_name():
                module.eval()

    def _step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model_ema = update_ema_variables(
            ema_model=self.model_ema, model=self.base_model, alpha_teacher=self.mt
        )

    def _target_outputs(self, target):
        self.base_model.train()
        outputs, _ = self.base_model.gnn(target.x, target.edge_index)
        self._set_bn_eval()
        return outputs

    def _replay_loss(self, memory_graphs):
        loss = 0.0
        for graph in memory_graphs:
            outputs, _ = self.base_model.gnn(graph.x, graph.edge_index)
            loss = loss + self.replay_weight * self.softmax_entropy(outputs, outputs)
        return loss

    def forward_full(self, target, i, epochs):
        outputs = self._target_outputs(target)
        loss = self.softmax_entropy(outputs, outputs) + self._replay_loss(self.memory)
        self._step(loss)

        if i < self.warmup_epochs:
            return outputs

        if i == epochs - 1:
            self.memory.append(graph_condense(self.args, target))
        return outputs

    def forward_no_replay(self, target, i, epochs):
        outputs = self._target_outputs(target)
        loss = self.softmax_entropy(outputs, outputs)
        self._step(loss)

        if i < self.warmup_epochs:
            return outputs

        if i == epochs - 1:
            # The condensed graph is still computed and stored, but it is not
            # replayed during adaptation.
            self.memory.append(graph_condense(self.args, target))
        return outputs

    def forward_no_condensation(self, target, i, epochs):
        outputs = self._target_outputs(target)
        loss = self.softmax_entropy(outputs, outputs) + self._replay_loss(self.memory)
        self._step(loss)

        if i < self.warmup_epochs:
            return outputs

        if i == epochs - 1:
            # Raw target graphs are stored instead of condensed memories.
            self.memory.append(
                Data(x=target.x.detach().clone(), edge_index=target.edge_index.detach().clone())
            )
        return outputs

    def softmax_entropy(self, x, x_ema):
        # L_Apt: entropy loss p log(p) + diversity regularisation.
        entropy = -(x_ema.softmax(1) * x.log_softmax(1)).sum(1).mean()
        mean_softmax = x.softmax(1).mean(0)
        entropy += self.entropy_weight * (mean_softmax * torch.log(mean_softmax)).sum()
        return entropy
