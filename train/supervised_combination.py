"""Supervised training of the logreg model."""
import torch
import torch.nn as nn
import numpy as np
from dataset import load


def create_random_fourier_features(X, n_components=1024, sigma=1, gamma=0.1):
    """Create random fourier features from the input data."""
    n_samples, n_features = X.shape
    np.random.seed(0)
    freqs = torch.FloatTensor(np.random.normal(size=(n_features, n_components))).to(
        X.device
    )
    offsets = torch.FloatTensor(
        np.random.uniform(0, 2 * np.pi, size=(n_components,))
    ).to(X.device)
    return (
        torch.sqrt(torch.tensor(2 / n_components)).to(X.device)
        * torch.cos(gamma * (X @ freqs) + offsets)
        / torch.sqrt(torch.tensor(sigma)).to(X.device)
    )


class LogRegEmbeddings(nn.Module):
    """Logistic regression combination model."""

    def __init__(
        self,
        ft_in,
        nb_classes,
        num_embeddings,
        activation="exp",
        individual_encoders=False,
        masking_alphas=None,
    ):
        super(LogRegEmbeddings, self).__init__()
        self.individual_encoders = individual_encoders

        if masking_alphas is not None:
            self.masking_alphas = (
                torch.from_numpy(np.asarray(masking_alphas)).float().cuda()
            )
        else:
            self.masking_alphas = None
        if not individual_encoders:
            self.fc = nn.Linear(ft_in, nb_classes)
        else:
            self.fc = nn.ModuleList(
                [nn.Linear(ft_in, nb_classes) for _ in range(num_embeddings)]
            )
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings, 1, 1))
        self.weight.data.fill_(0.1)

        for module in self.modules():
            self.weights_init(module)

        if activation == "exp":
            self.activation = torch.exp
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=0)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None

    def weights_init(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        """Forward pass."""
        if not self.individual_encoders:
            if self.activation is None:
                alphas = (
                    self.weight
                    if self.masking_alphas is None
                    else self.weight * self.masking_alphas.reshape(self.weight.shape)
                )
            else:
                alphas = (
                    self.activation(self.weight)
                    if self.masking_alphas is None
                    else self.activation(self.weight)
                    * self.masking_alphas.reshape(self.weight.shape)
                )
            outputs = torch.sum(alphas * seq, dim=0)

            ret = torch.log_softmax(self.fc(outputs), dim=-1)
        else:
            outputs = 0
            if self.activation is None:
                for i, seq_i in enumerate(seq):
                    outputs = outputs + self.weight[i] * self.fc[i](seq_i)
            else:
                for i, seq_i in enumerate(seq):
                    outputs = outputs + self.activation(self.weight[i]) * self.fc[i](
                        seq_i
                    )

            ret = torch.log_softmax(outputs, dim=-1)

        return ret


def post_combination(model, args):
    """Alpha combination supervised training for FiGURe."""
    print("Supervised Combination Training")
    adj, aug_adj_arr, feat, features_arr, labels, trainidx, validx, testidx = load(
        args["dataset"], options=args
    )

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(trainidx)
    idx_val = torch.LongTensor(validx)
    idx_test = torch.LongTensor(testidx)
    if torch.cuda.is_available():
        model.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    embeddings_list = []
    for i, aug_adj in enumerate(aug_adj_arr):
        if args["algorithm"] == "dgi":
            g1, g2 = adj, adj
        elif args["algorithm"] == "hassani":
            g1, g2 = adj, aug_adj
        elif args["algorithm"] == "figure":
            g1, g2 = aug_adj, aug_adj

        features = torch.FloatTensor(features_arr[i][np.newaxis])
        features = features.cuda()
        embeds = model.embed(features, g1, g2)
        embeddings_list.append(embeds)
    embeddings_list.append(model.embed_pre(features)[0])
    if args["gamma"] != 0:
        new_embeddings_list = []
        for embeds in embeddings_list:
            embeds = (
                create_random_fourier_features(
                    embeds[0],
                    n_components=512,
                    gamma=args["gamma"],
                )
                .unsqueeze(0)
                .cuda()
                .float()
            )
            new_embeddings_list.append(embeds)
        embeddings_list = new_embeddings_list

    for i, embeddings in enumerate(embeddings_list):
        args[f"embedding_dim_{i + 1}"] = embeddings.size()[-1]

    model = LogRegEmbeddings(
        args["embedding_dim_1"],
        torch.max(labels) + 1,
        len(embeddings_list),
        args["alpha_activation"],
        masking_alphas=[
            args[f"masking_alpha_{i}"] for i in range(len(embeddings_list))
        ],
    )

    embeds = embeddings_list
    n_epoch = 10000
    wd = args["logreg_weight_decay"]
    lr = args["logreg_lr"]
    lr_alphas = args["lr_alphas"]
    early_stop_patience = 10000

    train_embs = torch.stack([embed[0, trainidx] for embed in embeds]).cuda()
    val_embs = torch.stack([embed[0, validx] for embed in embeds]).cuda()
    test_embs = torch.stack([embed[0, testidx] for embed in embeds]).cuda()

    train_lbls = torch.tensor(labels[trainidx]).cuda()
    val_lbls = torch.tensor(labels[validx]).cuda()
    test_lbls = torch.tensor(labels[testidx]).cuda()

    model = model.cuda()
    opt = torch.optim.Adam(
        [
            {"params": model.fc.parameters(), "weight_decay": wd, "lr": lr},
            {"params": model.weight, "weight_decay": 0.0, "lr": lr_alphas},
        ]
    )
    xent = nn.CrossEntropyLoss()
    accs_val = []
    accs = []
    accs_train = []
    best_val_loss = float("inf")

    for epoch in range(n_epoch):
        model.train()
        opt.zero_grad()
        logits = model(train_embs)
        loss = 0
        loss = xent(logits, train_lbls)
        loss.backward()
        opt.step()

        with torch.no_grad():
            logits = model(val_embs)
            preds = torch.argmax(logits, dim=1)
            loss_val = xent(logits, val_lbls)
            acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            accs_val.append(acc.cpu().item() * 100)

            logits = model(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            accs.append(acc.cpu().item() * 100)

            logits = model(train_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == train_lbls).float() / train_lbls.shape[0]
            accs_train.append(acc.cpu().item() * 100)

            if loss_val < best_val_loss:
                best_val_loss = loss_val
                eval_epochs_since_improvement = 0
            else:
                eval_epochs_since_improvement += 1

            if eval_epochs_since_improvement >= early_stop_patience:
                print(
                    f"Stopping early after {epoch+1} epochs with no improvement in validation loss."
                )
                break

    result_dict = {
        "split_"
        + str(args["dataset_split"])
        + "_test_accuracy": accs[np.argmax(accs_val)],
        "split_" + str(args["dataset_split"]) + "_val_accuracy": np.max(accs_val),
    }
    print(result_dict)
    return result_dict
