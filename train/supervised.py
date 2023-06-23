"""Supervised training of the logreg model."""
import torch
import torch.nn as nn
import numpy as np
from dataset import load


def create_random_fourier_features(X, n_components=1024, sigma=1, gamma=0.1):
    """Create random fourier features from the input data."""
    np.random.seed(0)
    n_samples, n_features = X.shape
    freqs = np.random.normal(size=(n_features, n_components))
    offsets = np.random.uniform(0, 2 * np.pi, size=(n_components,))
    Z = (
        np.sqrt(2 / n_components)
        * np.cos(gamma * (X @ freqs) + offsets)
        / np.sqrt(sigma)
    )
    return Z


class LogReg(nn.Module):
    """Logistic regression model."""

    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for module in self.modules():
            self.weights_init(module)

    def weights_init(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        """Forward pass."""
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret


def train_supervised(
    model,
    options=None,
):
    """Train the model using supervised learning."""
    adj, aug_adj_arr, feat, features_arr, labels, idx_train, idx_val, idx_test = load(
        options["dataset"], options=options
    )

    nb_classes = np.unique(labels).shape[0]
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    if torch.cuda.is_available():
        model.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    xent = nn.CrossEntropyLoss()
    result_dict = {}
    all_accs_val = []

    for i, aug_adj in enumerate(aug_adj_arr):
        if options["algorithm"] == "dgi":
            g1, g2 = adj, adj
        elif options["algorithm"] == "mvgrl":
            g1, g2 = adj, aug_adj
        elif options["algorithm"] == "figure":
            g1, g2 = aug_adj, aug_adj
        features = torch.FloatTensor(features_arr[i]).unsqueeze(0)
        features = features.cuda()
        embeds = model.embed(features, g1, g2)

    if options["gamma"] != 0:
        embeds = (
            torch.tensor(
                create_random_fourier_features(
                    embeds[0].cpu().numpy(),
                    n_components=512,
                    gamma=options["gamma"],
                )
            )
            .unsqueeze(0)
            .cuda()
            .float()
        )
    train_embs = embeds[0, idx_train]
    train_lbls = labels[idx_train]
    val_lbls = labels[idx_val]
    test_lbls = labels[idx_test]
    wd = options["logreg_weight_decay"]
    hid_units = embeds.shape[2]

    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=options["logreg_lr"], weight_decay=wd)
    log.cuda()
    best_accs_val_inner = -1
    accs_inner = -1
    preds_inner = -1
    for epoch in range(options["logreg_epochs"]):
        log.train()
        opt.zero_grad()
        logits = log(train_embs)
        loss = xent(logits, train_lbls)
        loss.backward()
        opt.step()
        with torch.no_grad():
            log.eval()
            logits = log(embeds[0, :])
            preds = torch.argmax(logits, dim=1)
            accval = (
                100 * torch.sum(preds[idx_val] == val_lbls).float() / val_lbls.shape[0]
            )
            if accval > best_accs_val_inner:
                best_accs_val_inner = accval
                accs_inner = (
                    100
                    * torch.sum(preds[idx_test] == test_lbls).float()
                    / test_lbls.shape[0]
                )
                preds_inner = preds

    preds = preds_inner
    accs = accs_inner
    accs_val = best_accs_val_inner
    print("val:", accs_val.item())
    print("test: ", accs.item())
    result_dict.update(
        {
            "split_" + str(options["dataset_split"]) + "_test_accuracy": accs.item(),
            "split_" + str(options["dataset_split"]) + "_val_accuracy": accs_val.item(),
        }
    )
    all_accs_val.append(accs_val.mean().item())
    return result_dict
