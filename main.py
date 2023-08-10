"""This is the main file for running the experiments. 
It contains the hyperparameter search space."""
import argparse
import copy
import numpy as np
import torch
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from utils import set_seed
from train.unsupervised import train_unsupervised
from train.supervised import train_supervised as train_node_supervised_1
from train.supervised_combination import post_combination

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cora", type=str)  #
parser.add_argument("--seed", default=5, type=int)  #
parser.add_argument("--sample_size", default=2000, type=int)  #
parser.add_argument("--lr", default=0.001, type=float)  #
parser.add_argument("--hid_units", default=512, type=int)  #
parser.add_argument("--logreg_weight_decay", default=0, type=float)  #
parser.add_argument("--logreg_lr", default=0.001, type=float)  #
parser.add_argument("--logreg_epochs", default=10000, type=int)  #
parser.add_argument("--batch_size", default=1, type=int)  #
parser.add_argument("--dataset_split", default=-1, type=int)  #
parser.add_argument("--output_dir", default="./output", type=str)  #
parser.add_argument("--data_dir", default="datasets", type=str)  #
parser.add_argument("--algorithm", default="mvgrl", type=str)  #
parser.add_argument(
    "--sparse", default=False, type=lambda x: (str(x).lower() == "true")
)  #
parser.add_argument("--n_trials_unsup", default=20, type=int)  #
parser.add_argument("--n_trials_sup", default=60, type=int)  #
parser.add_argument("--alpha_masks", default="-1", type=str)  #
parser.add_argument("--lr_alphas", default=0.001, type=float)  #
parser.add_argument("--alpha_activation", default="none", type=str)  #
parser.add_argument("--gamma", default=0, type=float)  #


# get options
options = vars(parser.parse_args())

if options["algorithm"] == "mvgrl":
    options["augmentation_type"] = "diff"
    options["augmentation_quantity"] = 0
    options["augmentation_index"] = 0
    options["augmentation_all"] = False

if options["algorithm"] == "figure":
    options["augmentation_type"] = "fsgnn"
    options["augmentation_quantity"] = 2
    options["augmentation_index"] = 0
    options["augmentation_all"] = True


if (
    options["dataset"] == "texas"
    or options["dataset"] == "wisconsin"
    or options["dataset"] == "cornell"
):
    options[
        "sample_size"
    ] = 0  ##Since these graphs are small, use the whole graph instead of sampling

if options["dataset_split"] == -1:
    TOTAL_SPLITS = 10
    if options["dataset"] == "ogbn_arxiv":
        TOTAL_SPLITS = 1
    elif options["dataset"] == "arxiv-year":
        TOTAL_SPLITS = 5
    elif options["dataset"] == "twitch-gamer":
        TOTAL_SPLITS = 5

    dataset_splits = list(range(TOTAL_SPLITS))
    options["dataset_split"] = 0
else:
    dataset_splits = [options["dataset_split"]]

print(options)

train_node_supervised = (
    post_combination if options["augmentation_all"] else train_node_supervised_1
)

param_space = {
    "batch_size": tune.choice([2, 4, 8])
    if options["batch_size"] == -1
    else options["batch_size"],
    "lr": tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    if options["lr"] == -1
    else options["lr"],
    "sample_size": tune.choice([1500, 1750, 2000, 2250])
    if options["sample_size"] == -1
    else options["sample_size"],
    "hid_units": tune.choice([32, 64, 128, 256, 512])
    if options["hid_units"] == -1
    else options["hid_units"],
    "logreg_weight_decay": tune.choice(
        [
            0,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            0.5,
            1,
            3,
        ]
    )
    if options["logreg_weight_decay"] == -1
    else options["logreg_weight_decay"],
    "logreg_epochs": 10000
    if options["logreg_epochs"] == -1
    else options["logreg_epochs"],
    "logreg_lr": tune.choice([0.01, 0.001, 0.0001, 0.1, 0.015, 0.0015, 1, 0.5, 2, 1e-5])
    if options["logreg_lr"] == -1
    else options["logreg_lr"],
    "lr_alphas": tune.choice([0.01, 0.001, 0.0001, 0.1, 0.015, 0.0015, 1, 0.5, 2, 1e-5])
    if options["lr_alphas"] == -1
    else options["lr_alphas"],
    "alpha_activation": tune.choice(["exp", "none"])
    if options["alpha_activation"] == "-1"
    else options["alpha_activation"],
    "gamma": tune.choice([0.1 * n for n in range(1, 12)])
    if options["gamma"] == -1
    else options["gamma"],
}
masking_alphas = {
    f"masking_alpha_{i}": tune.choice([0, 1])
    if options["alpha_masks"] == "-1"
    else int(options["alpha_masks"][i])
    for i in range(2 + options["augmentation_quantity"])
}

param_space.update(masking_alphas)
if options["alpha_masks"] != "-1":
    options.update(masking_alphas)
###Adding 2 because including XW
unsupervised_param_names = [
    "lr",
    "hid_units",
    "batch_size",
    "sample_size",
]
supervised_param_names = ["logreg_weight_decay", "logreg_epochs", "logreg_lr", "gamma"]
supervised_alpha_combination_param_names = [
    "logreg_weight_decay",
    "logreg_epochs",
    "logreg_lr",
    "lr_alphas",
    "alpha_activation",
    "gamma",
] + [f"masking_alpha_{i}" for i in range(2 + options["augmentation_quantity"])]


unsupervised_param_space = {key: param_space[key] for key in unsupervised_param_names}
supervised_param_space = {key: param_space[key] for key in supervised_param_names}
supervised_alpha_combination_param_space = {
    key: param_space[key] for key in supervised_alpha_combination_param_names
}
print("Unsupervised Training")

new_ops = copy.deepcopy(
    options
)  # new_ops contaains the hparam options, and in case -1 is specified, it will be replaced by the best hparam
if any(
    (value == -1 or value == "-1")
    for value in [options[key] for key in unsupervised_param_names]
):

    def train_unsupervised_wrapper(params):
        """Unsupervised Hyperparameter Optimization"""
        tune.utils.wait_for_gpu(target_util=0.2)
        set_seed(options["seed"])
        options2 = {}
        options2.update(options)
        options2.update(params)
        return train_unsupervised(options2["dataset"], options=options2)[0]

    results = tune.run(
        train_unsupervised_wrapper,
        mode="min",
        search_alg=ConcurrencyLimiter(
            OptunaSearch(seed=options["seed"]), max_concurrent=32
        ),
        num_samples=options["n_trials_unsup"],
        config=unsupervised_param_space,
        resources_per_trial={"cpu": 1, "gpu": 1},
        raise_on_failed_trial=False,
    )
    new_ops.update(results.get_best_config())

set_seed(options["seed"])
loss, model = train_unsupervised(verbose=True, options=new_ops)
model.to("cpu")
torch.cuda.empty_cache()
unsupervised_result_dict = {"unsupervised_loss": loss}
print(unsupervised_result_dict)


print("Supervised Training")


supervised_result_dict = {}
options_copy = {}
options_copy.update(options)
for split in dataset_splits:
    print("Split", split)
    options_copy["dataset_split"] = split
    if any(
        (
            (value == -1 or value == "-1")
            for value in [options_copy[key] for key in supervised_param_names]
        )
    ) or (
        options["augmentation_all"]
        and (
            options["alpha_masks"] == "-1"
            or options["lr_alphas"] == -1
            or options["alpha_activation"] == "-1"
        )
    ):

        def train_supervised_wrapper(params):
            """Supervised Hyperparameter Optimization"""
            tune.utils.wait_for_gpu(target_util=0.2)
            set_seed(options_copy["seed"])
            options2 = {}
            options2.update(options_copy)
            options2.update(params)
            print("params:", options_copy)
            trial_results = train_node_supervised(
                model,
                options2,
            )
            return trial_results[
                "split_" + str(options_copy["dataset_split"]) + "_val_accuracy"
            ]

        results = tune.run(
            train_supervised_wrapper,
            mode="max",
            search_alg=ConcurrencyLimiter(OptunaSearch(), max_concurrent=32),
            num_samples=options["n_trials_sup"],
            config=(
                supervised_alpha_combination_param_space
                if options["augmentation_all"]
                else supervised_param_space
            ),
            resources_per_trial={"cpu": 1, "gpu": 1},
            raise_on_failed_trial=False,
        )
        options_copy.update(results.get_best_config())

    seed = options["seed"]
    set_seed(seed)
    supervised_result_dict_one_split = train_node_supervised(
        model,
        options_copy,
    )
    supervised_result_dict.update(supervised_result_dict_one_split)

all_split_results = np.array(
    [
        supervised_result_dict["split_" + str(spl) + "_test_accuracy"]
        for spl in dataset_splits
    ]
)
supervised_result_dict["mean_test_accuracy"] = np.mean(all_split_results)
supervised_result_dict["std_test_accuracy"] = np.std(all_split_results)
results_dict = {**unsupervised_result_dict, **supervised_result_dict}
print("Results dict", results_dict)

print("Note that these results are for the following data splits:", dataset_splits)
