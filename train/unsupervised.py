"""This file contains the code for training the unsupervised model."""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from utils import sparse_mx_to_torch_sparse_tensor
from dataset import load
from scipy.sparse import issparse


# Borrowed from https://github.com/PetarV-/DGI
class GCN(nn.Module):
    """Graph convolutional network"""

    def __init__(
        self,
        in_ft,
        out_ft,
        bias=True,
    ):
        super(GCN, self).__init__()
        self.act = nn.PReLU()
        self.fc = nn.Linear(in_ft, out_ft, bias=bias)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False, embed_pre=False):
        """Forward pass"""
        out = self.fc(seq)
        if embed_pre:
            return self.act(out)
        if sparse:
            outs = []
            for adj_in, out_in in zip(adj, out):
                outs.append(torch.spmm(adj_in, torch.squeeze(out_in, 0)))
            out = torch.stack(outs, dim=0)
        else:
            out = torch.bmm(adj, out)
        return self.act(out)


# Borrowed from https://github.com/PetarV-/DGI
class Readout(nn.Module):
    """Graph Readout function"""

    def forward(self, seq):
        """Forward pass"""
        return torch.mean(seq, 1)


# Borrowed from https://github.com/PetarV-/DGI
class Discriminator(nn.Module):
    """Discriminator model"""

    def __init__(self, n_h, algorithm):
        super(Discriminator, self).__init__()
        self.algorithm = algorithm
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for module in self.modules():
            self.weights_init(module)

    def weights_init(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c_1, c_2, h_1, h_2, h_3, h_4):
        """forward pass"""
        if self.algorithm == "mvgrl":
            c_x1 = torch.unsqueeze(c_1, 1)
            c_x1 = c_x1.expand_as(h_1).contiguous()
            c_x2 = torch.unsqueeze(c_2, 1)
            c_x2 = c_x2.expand_as(h_2).contiguous()

            # positive
            sc_1 = torch.squeeze(self.f_k(h_2, c_x1), 2)
            sc_2 = torch.squeeze(self.f_k(h_1, c_x2), 2)

            # negetive
            sc_3 = torch.squeeze(self.f_k(h_4, c_x1), 2)
            sc_4 = torch.squeeze(self.f_k(h_3, c_x2), 2)

            logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
            return logits

        elif self.algorithm == "dgi" or self.algorithm == "figure":
            c_x1 = torch.unsqueeze(c_1, 1)
            c_x1 = c_x1.expand_as(h_1).contiguous()
            # positive
            sc_1 = torch.squeeze(self.f_k(h_1, c_x1), 2)
            # negetive
            sc_2 = torch.squeeze(self.f_k(h_3, c_x1), 2)
            logits = torch.cat((sc_1, sc_1, sc_2, sc_2), 1)
            return logits


class Model(nn.Module):
    """Model class"""

    def __init__(self, n_in, n_h, algorithm="dgi", sparse=False):
        super(Model, self).__init__()
        self.algorithm = algorithm
        self.sparse = sparse
        self.gcn1 = GCN(
            n_in,
            n_h,
        )

        if algorithm == "mvgrl":
            self.gcn2 = GCN(
                n_in,
                n_h,
            )

        self.read = Readout()
        self.global_representation_activation = nn.Sigmoid()
        self.disc = Discriminator(n_h, algorithm)

    def forward(self, seq1, seq2, adj, diff):
        """Forward pass"""
        if self.algorithm == "dgi" or self.algorithm == "figure":
            h_1 = self.gcn1(seq1, adj, self.sparse)
            c_1 = self.read(h_1)
            c_1 = self.global_representation_activation(c_1)
            h_3 = self.gcn1(seq2, adj, self.sparse)
            ret = self.disc(c_1, None, h_1, None, h_3, None)
            return ret

        elif self.algorithm == "mvgrl":
            h_1 = self.gcn1(seq1, adj, self.sparse)
            c_1 = self.read(h_1)
            c_1 = self.global_representation_activation(c_1)
            h_2 = self.gcn2(seq1, diff, self.sparse)
            c_2 = self.read(h_2)
            c_2 = self.global_representation_activation(c_2)
            h_3 = self.gcn1(seq2, adj, self.sparse)
            h_4 = self.gcn2(seq2, diff, self.sparse)
            ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4)

            return ret

    def embed(self, seq, adj, diff):
        """Fetch the embeddings"""
        if self.sparse:
            if self.algorithm == "mvgrl":
                raise NotImplementedError
            #### Slide a windows over adj to prevent OOM
            h_batch = []
            sample_size = min((5000, adj.shape[0]))
            for i in range(int(adj.shape[0] / sample_size + 1)):
                if i == int(adj.shape[0] / sample_size):
                    adj2 = adj[i * sample_size :, :]
                else:
                    adj2 = adj[i * sample_size : (i + 1) * sample_size, :]
                with torch.no_grad():
                    h_1 = self.gcn1(
                        seq,
                        sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj2))
                        .unsqueeze(0)
                        .to(seq.device),
                        True,
                    )
                    h_batch.append(h_1)
            return torch.cat(h_batch, 1).to_dense()
        else:
            adj = adj.todense() if issparse(adj) else adj
            adj = torch.tensor(adj).unsqueeze(0)
            if torch.cuda.is_available():
                adj = adj.cuda()
            h_1 = self.gcn1(seq, adj, self.sparse)
            if self.algorithm == "mvgrl":
                diff = diff.todense() if issparse(diff) else diff
                diff = torch.tensor(diff).float().unsqueeze(0)
                if torch.cuda.is_available():
                    diff = diff.cuda()
                h_2 = self.gcn2(seq, diff, self.sparse)
                return (h_1 + h_2).detach()
            return (h_1 + h_1).detach()

    def embed_pre(self, seq):
        """Fetch the embeddings using the identity filter"""
        if self.algorithm == "mvgrl" or self.algorithm == "dgi":
            raise NotImplementedError
        with torch.no_grad():
            h_1 = self.gcn1(
                seq,
                None,
                True,
                embed_pre=True,
            )
        return h_1, None


def train_unsupervised(verbose=False, options=None):
    """Unsupervised training loop"""
    nb_epochs = 30000
    patience = 20
    if options["dataset"]=="ogbn_arxiv" or options["dataset"]=="arxiv-year":
        patience=500
    lr = options["lr"]
    l2_coef = 0
    hid_units = options["hid_units"]
    sparse = options["sparse"]

    (
        adj,
        aug_adj_list,
        feat,
        _,
        _,
        _,
        _,
        _,
    ) = load(options["dataset"], options=options)

    if not sparse:
        adj = torch.FloatTensor(adj.todense())
        if issparse(aug_adj_list[0]):
            aug_adj_list = [aug_adj.todense() for aug_adj in aug_adj_list]
        aug_adj_list = [torch.FloatTensor(aug_adj) for aug_adj in aug_adj_list]
        if torch.cuda.is_available():
            adj = adj.cuda()
            aug_adj_list = [aug_adj.cuda() for aug_adj in aug_adj_list]

    ft_size = feat.shape[1]
    sample_size = options["sample_size"]
    if sample_size == 0:
        options["batch_size"] = 1
        sample_size = adj.shape[-1]

    batch_size = options["batch_size"]

    lbl_1 = torch.ones(batch_size, sample_size * 2)
    lbl_2 = torch.zeros(batch_size, sample_size * 2)
    lbl_pos_neg = torch.cat((lbl_1, lbl_2), 1)
    lbl_neg_neg = torch.cat((lbl_2, lbl_2), 1)

    model = Model(
        ft_size,
        hid_units,
        algorithm=options["algorithm"],
        sparse=options["sparse"],
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        model.cuda()
        lbl_pos_neg = lbl_pos_neg.cuda()
        lbl_neg_neg = lbl_neg_neg.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    cnt_wait = 0
    best = 1e9

    for epoch in range(nb_epochs):
        loss_per_epoch = []

        for ind, aug_graph in enumerate(aug_adj_list):
            idx = np.random.randint(
                0, aug_graph.shape[-1] - sample_size + 1, batch_size
            )
            if options["algorithm"] == "dgi":
                graph1, graph2 = adj, adj
            elif options["algorithm"] == "mvgrl":
                graph1, graph2 = adj, aug_graph
            elif options["algorithm"] == "figure":
                graph1, graph2 = aug_graph, aug_graph

            ba, bd, bf = [], [], []
            for i in idx:
                ba.append(graph1[i : i + sample_size, i : i + sample_size])
                bd.append(graph2[i : i + sample_size, i : i + sample_size])
                bf.append(feat[i : i + sample_size])

            if sparse:
                ba = torch.stack(
                    [sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(m)) for m in ba],
                    dim=0,
                )
                bd = torch.stack(
                    [sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(m)) for m in bd],
                    dim=0,
                )
            else:
                ba = torch.stack(ba)
                bd = torch.stack(bd)

            bf = np.array(bf).reshape(batch_size, sample_size, ft_size)
            bf = torch.FloatTensor(bf)
            idx = np.random.permutation(sample_size)
            shuf_fts = bf[:, idx, :]

            if torch.cuda.is_available():
                bf = bf.cuda()
                ba = ba.cuda()
                bd = bd.cuda()
                shuf_fts = shuf_fts.cuda()

            model.train()
            logits = model(
                bf,
                shuf_fts,
                ba,
                bd,
            )

            optimiser.zero_grad()
            loss = b_xent(logits, lbl_pos_neg)
            loss.backward()
            optimiser.step()
            loss_per_epoch.append(loss.item())

        loss = np.mean(loss_per_epoch)
        if verbose:
            print("Epoch: {0}, Loss: {1:0.4f}".format(epoch, loss))

        if loss < best:
            best = loss
            cnt_wait = 0
            torch.save(model.state_dict(), "model.pkl")
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            if verbose:
                print("Early stopping!")
            break

    model.load_state_dict(torch.load("model.pkl"))
    return best, model
