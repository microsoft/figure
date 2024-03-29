import numpy as np
import networkx as nx
import torch
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp
import random


def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    try:
        a = nx.convert_matrix.to_numpy_array(graph)
    except:
        a = graph
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(np.absolute(a), 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1


def compute_heat(graph: nx.Graph, t=5, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])
    d = np.diag(np.sum(a, 1))
    return np.exp(t * (np.matmul(a, inv(d)) - 1))


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj, self_loop=True):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    return normalize_sparse_graph(adj, -0.5, -0.5)
    # rowsum = np.array(np.abs(adj).sum(1))
    # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # return (
    #     adj.dot(d_mat_inv_sqrt)
    #     .transpose()
    #     .dot(d_mat_inv_sqrt)
    #     .tocoo()
    #     .astype(np.float32)
    # )


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_sparse_graph(graph, gamma, beta):
    """
    Utility function for normalizing sparse graphs.
    return Dr^gamma x graph x Dc^beta
    """
    b_graph = graph.tocsr().copy().astype(np.float32)
    r_graph = b_graph.copy()
    c_graph = b_graph.copy()
    row_sums = []
    for i in range(graph.shape[0]):
        row_sum = r_graph.data[r_graph.indptr[i] : r_graph.indptr[i + 1]].sum()
        if row_sum == 0:
            row_sums.append(0.0)
        else:
            row_sums.append(row_sum**gamma)

    c_graph = c_graph.tocsc()
    col_sums = []
    for i in range(graph.shape[1]):
        col_sum = c_graph.data[c_graph.indptr[i] : c_graph.indptr[i + 1]].sum()

        if col_sum == 0:
            col_sums.append(0.0)
        else:
            col_sums.append(col_sum**beta)

    for i in range(graph.shape[0]):
        if row_sums[i] != 0:
            b_graph.data[r_graph.indptr[i] : r_graph.indptr[i + 1]] *= row_sums[i]

    b_graph = b_graph.tocsc()
    for i in range(graph.shape[1]):
        if col_sums[i] != 0:
            b_graph.data[c_graph.indptr[i] : c_graph.indptr[i + 1]] *= col_sums[i]
    return b_graph


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
