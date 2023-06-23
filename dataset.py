"""Data loader for node classification task"""
import os
import os.path as osp
import pickle as pkl
import scipy.sparse as sp
import numpy as np
import networkx as nx
import torch
import dgl
from utils import compute_ppr, normalize_adj


def index_to_mask(index, size):
    """Convert index to mask"""
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def load_dataset_from_file(
    name,
    options=None,
):
    """Load dataset from file"""
    split = options["dataset_split"]

    data_dir = osp.join(
        os.path.dirname(os.path.realpath(__file__)),
        options["data_dir"]
    )

    if name in [
        "roman_empire",
        "amazon_ratings",
        "minesweeper",
        "tolokers",
        "questions",
    ]:
        data = np.load(osp.join(data_dir, f'{name.replace("-", "_")}.npz'))
        dataset_data = data

        node_features = data["node_features"]
        edges = data["edges"]
        n = len(node_features)
        adj_matrix = [[0] * n for _ in range(n)]
        for u, v in edges:
            adj_matrix[int(u)][int(v)] = 1
            adj_matrix[int(v)][int(u)] = 1
        adj_matrix = np.array(adj_matrix)
        edges = np.nonzero(adj_matrix)
        labels = torch.tensor(data["node_labels"])
        train_mask = torch.LongTensor(np.array(dataset_data["train_masks"][split]) * 1)
        val_mask = torch.LongTensor(np.array(dataset_data["val_masks"][split]) * 1)
        test_mask = torch.LongTensor(np.array(dataset_data["test_masks"][split]) * 1)

        g = dgl.graph((edges[0], edges[1]))
        g.ndata["feat"] = torch.tensor(node_features)
        g.ndata["label"] = torch.tensor(labels, dtype=torch.long)
        g.ndata["train_mask"] = train_mask
        g.ndata["val_mask"] = val_mask
        g.ndata["test_mask"] = test_mask

        g = dgl.remove_self_loop(g)
        src, dst = g.edges()
        g.add_edges(dst, src)
        dgl_graph = g

    elif name in ["twitch-gamer", "arxiv-year"]:
        dataset_data = pkl.load(open(osp.join(data_dir, name + ".pkl"), "rb"))
        dataset = dataset_data["dataset"][0]
        labels = dataset_data["dataset"][1]
        splits = dataset_data["split_idx_lst"]
        edge_index = dataset["edge_index"]
        features = dataset["node_feat"]
        dgl_graph = dgl.DGLGraph()
        dgl_graph.add_nodes(features.shape[0])
        dgl_graph.add_edges(edge_index[0, :], edge_index[1, :])
        dgl_graph.ndata["feat"] = torch.tensor(features, dtype=torch.float)
        dgl_graph.ndata["label"] = torch.squeeze(labels, dim=-1)
        dgl_graph.ndata["train_mask"] = index_to_mask(
            splits[split]["train"].numpy(), size=features.shape[0]
        )
        dgl_graph.ndata["val_mask"] = index_to_mask(
            splits[split]["valid"].numpy(), size=features.shape[0]
        )
        dgl_graph.ndata["test_mask"] = index_to_mask(
            splits[split]["test"].numpy(), size=features.shape[0]
        )

    else:
        dataset_data = pkl.load(open(osp.join(data_dir, name + ".pkl"), "rb"))

        edges = dataset_data["sym_adj"].nonzero()
        features = dataset_data["X"]
        dgl_graph = dgl.DGLGraph()
        dgl_graph.add_nodes(features.shape[0])
        dgl_graph.add_edges(edges[0], edges[1])
        dgl_graph.ndata["feat"] = torch.tensor(features, dtype=torch.float)
        dgl_graph.ndata["label"] = torch.tensor(
            dataset_data["labels"].argmax(1), dtype=torch.long
        )
        dgl_graph.ndata["train_mask"] = index_to_mask(
            dataset_data["split_" + str(split)]["train_ids"],
            size=dataset_data["X"].shape[0],
        )
        dgl_graph.ndata["val_mask"] = index_to_mask(
            dataset_data["split_" + str(split)]["val_ids"],
            size=dataset_data["X"].shape[0],
        )
        dgl_graph.ndata["test_mask"] = index_to_mask(
            dataset_data["split_" + str(split)]["test_ids"],
            size=dataset_data["X"].shape[0],
        )

    return dgl.remove_self_loop(dgl_graph)


def load(dataset, options):
    """Load dataset"""
    dgl_graph = load_dataset_from_file(dataset, options=options)
    augmentations_adj_list = []
    augmentations_feat_list = []
    nx_graph = dgl_graph.to_networkx()

    if options["algorithm"] == "mvgrl":
        augmentations_adj_list.append(
            np.array(
                compute_ppr(
                    np.array(nx.adjacency_matrix(dgl_graph.to_networkx()).todense()),
                    0.2,
                )
            )
        )
        augmentations_feat_list.append(dgl_graph.ndata["feat"].numpy())
    elif options["algorithm"] == "figure":
        augmentation_quantity = options["augmentation_quantity"]
        n_adj = normalize_adj(nx.adjacency_matrix(nx_graph), self_loop=True)
        n_adj_powers = n_adj
        n_adj_powers_arr = [n_adj]
        for i in range(1, augmentation_quantity + 1):
            n_adj_powers = n_adj_powers @ n_adj
            n_adj_powers_arr.append(n_adj_powers)

        for k in range(augmentation_quantity + 1):
            adj = n_adj_powers_arr[k]
            augmentations_adj_list.append(adj)
            augmentations_feat_list.append(dgl_graph.ndata["feat"].numpy())

    train_mask = dgl_graph.ndata["train_mask"].numpy()
    val_mask = dgl_graph.ndata["val_mask"].numpy()
    test_mask = dgl_graph.ndata["test_mask"].numpy()
    feat = dgl_graph.ndata["feat"].numpy()

    adj = normalize_adj(sp.csr_matrix(nx.adjacency_matrix(nx_graph)), self_loop=True)
    idx_train = np.argwhere(train_mask * 1 == 1).reshape(-1)
    idx_val = np.argwhere(val_mask * 1 == 1).reshape(-1)
    idx_test = np.argwhere(test_mask * 1 == 1).reshape(-1)
    labels = dgl_graph.ndata["label"].numpy()
    return (
        adj,
        augmentations_adj_list,
        feat,
        augmentations_feat_list,
        labels,
        idx_train,
        idx_val,
        idx_test,
    )
