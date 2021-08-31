"""
Training loop for prototypical network over mini imagenet

Training script originally adapted from pytorch-meta:
https://github.com/tristandeleu/pytorch-meta/blob/8e739fa7b2d572c2044ca63215a99d0d3315bce5/examples/protonet/train.py#L1
"""
import argparse
import os
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

import sys

# This is needed to make script work even when inside package and we can't change the python path variable:

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from mtm.models.protonet.jsd import jsd_loss
from mtm.single_client.omniglot.omniglot import Omniglot
from mtm.util.early_stopper import EarlyStopper
from mtm.single_client.mini_imagenet.mini_imagenet import MiniImagenet
from mtm.models.protonet.model import (
    get_prototypes,
    prototypical_loss,
    PrototypicalNetwork,
    get_protonet_accuracy,
    SUPPORTED_POOLING_LAYERS,
)
from mtm.util.metrics import ci95

logger = logging.getLogger(__name__)

SUPPORTED_DATASETS = {"mini-imagenet", "omniglot"}
MINI_IMAGENET_TEST_SHOTS = 15
OMNIGLOT_TEST_SHOTS = 5


def eval_meta_test_dataset(model, meta_test_dataset, args: argparse.PARSER):
    if args.dataset == "mini-imagenet":
        return eval_mini_imagenet(model, meta_test_dataset, args)
    elif args.dataset == "omniglot":
        return eval_omniglot(model, meta_test_dataset, args)
    else:
        raise NotImplementedError(f"{args.dataset} not in {SUPPORTED_DATASETS}")


def eval_mini_imagenet(
    model: nn.Module, dataset: MiniImagenet, args: argparse.PARSER
) -> Tuple[float, float]:
    """Returns meta-test-set accuracy and confidence intervals as percentages"""
    model.eval()
    test_episodes = 600  # Following prototypical networks paper
    meta_test_test_set_accuracies = []
    with torch.no_grad():
        for i in range(test_episodes):
            batch = dataset.sample_meta_batch(
                batch_size=args.batch_size, meta_split="test", sample_k_value=False
            )
            train_inputs, train_targets = batch["train"]
            train_inputs = train_inputs.to(device=args.device)
            train_embeddings = model(train_inputs)

            test_inputs, test_targets = batch["test"]
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)
            test_embeddings = model(test_inputs)

            prototypes = get_prototypes(
                train_embeddings, dataset.n_classes_per_task, dataset.k_shots_per_class
            )
            accuracy = get_protonet_accuracy(
                prototypes, test_embeddings, test_targets, jsd=args.jsd
            )
            print(
                "Meta-test test-set accuracy at iteration {}: {}".format(
                    i, accuracy.item()
                )
            )
            meta_test_test_set_accuracies.append(accuracy.item())

    mean_acc = np.mean(meta_test_test_set_accuracies)
    ci = ci95(meta_test_test_set_accuracies)
    print("Meta-test set accuracy {}+-{}%".format(mean_acc * 100, ci * 100))
    model.train()

    return mean_acc * 100, ci * 100


def eval_omniglot(model: nn.Module, dataset: Omniglot, args) -> Tuple[float, float]:
    """Returns meta-test-set accuracy and confidence intervals as percentages"""
    model.eval()
    test_episodes = 1000  # Following prototypical networks paper
    meta_test_test_set_accuracies = []
    with torch.no_grad():
        for i in range(test_episodes):
            batch = dataset.sample_meta_batch(
                batch_size=args.batch_size,
            )
            train_inputs, train_targets = batch["train"]
            train_inputs = train_inputs.to(device=args.device)
            train_embeddings = model(train_inputs)

            test_inputs, test_targets = batch["test"]
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)
            test_embeddings = model(test_inputs)

            prototypes = get_prototypes(
                train_embeddings, dataset.n_classes_per_task, dataset.k_shots_per_class
            )
            accuracy = get_protonet_accuracy(
                prototypes, test_embeddings, test_targets, jsd=args.jsd
            )
            print(
                "Meta-test test-set accuracy at iteration {}: {}".format(
                    i, accuracy.item()
                )
            )
            meta_test_test_set_accuracies.append(accuracy.item())

    mean_acc = np.mean(meta_test_test_set_accuracies)
    ci = ci95(meta_test_test_set_accuracies)
    print("Meta-test set accuracy {}+-{}%".format(mean_acc * 100, ci * 100))
    model.train()

    return mean_acc * 100, ci * 100


def eval_meta_val_set(
    model: PrototypicalNetwork,
    meta_val_dataset: MiniImagenet,
    args,
    n_eval_batches: int = 600,
):
    model.eval()
    accuracies, losses = [], []
    for i in range(n_eval_batches):
        val_batch = meta_val_dataset.sample_meta_batch(
            batch_size=args.batch_size, meta_split="val", sample_k_value=False
        )
        train_inputs, train_targets = val_batch["train"]
        train_inputs = train_inputs.to(device=args.device)
        test_inputs, test_targets = val_batch["test"]
        test_inputs = test_inputs.to(device=args.device)
        test_targets = test_targets.to(device=args.device)

        train_embeddings = model(train_inputs)
        test_embeddings = model(test_inputs)
        prototypes = get_prototypes(
            train_embeddings,
            meta_val_dataset.n_classes_per_task,
            meta_val_dataset.k_shots_per_class,
        )  # -> [b, n, features]
        loss = prototypical_loss(prototypes, test_embeddings, test_targets)
        accuracy = get_protonet_accuracy(
            prototypes, test_embeddings, test_targets, jsd=args.jsd
        )
        accuracies.append(accuracy)
        losses.append(loss)
    accuracy = torch.mean(torch.stack(accuracies))
    loss = torch.mean(torch.stack(losses))
    model.train()
    return accuracy, loss


def train(args):
    assert args.data_dir is not None

    if args.dataset == "omniglot":
        in_channels = 1
        Dataset = Omniglot
        test_shots = OMNIGLOT_TEST_SHOTS
    elif args.dataset == "mini-imagenet":
        in_channels = 3
        Dataset = MiniImagenet
        test_shots = MINI_IMAGENET_TEST_SHOTS
    else:
        raise NotImplementedError(
            f"{args.dataset} not supported. --dataset must be in {SUPPORTED_DATASETS}"
        )

    if args.l2_weight_decay:
        weight_decay = 0.0005
    else:
        weight_decay = 0.0
    if (
        args.prototype_normal_std_noise is not None
        and args.prototype_normal_std_noise != 0
    ):
        prototype_normal_std_noise = args.prototype_normal_std_noise
    else:
        prototype_normal_std_noise = None

    sample_k_value = args.sample_k_value

    eval_interval = args.num_batches // args.n_val_during_training
    eval_interval = max(eval_interval, 1)
    train_num_ways = (
        args.num_ways if args.train_num_ways is None else args.train_num_ways
    )
    dataset = Dataset(
        args.data_dir,
        train_num_ways,
        args.num_shots,
        test_shots_per_class=test_shots,
        meta_split="train",
        fed_recon=args.fed_recon,
    )
    if args.dataset == "mini-imagenet":
        if args.fed_recon:
            meta_val_dataset = None
        else:
            meta_val_dataset = Dataset(
                args.data_dir,
                args.num_ways,
                args.num_shots,
                test_shots_per_class=test_shots,
                meta_split="val",
                fed_recon=args.fed_recon,
            )
    else:
        meta_val_dataset = None
    meta_test_dataset = Dataset(
        args.data_dir,
        args.num_ways,
        args.num_shots,
        test_shots_per_class=test_shots,
        meta_split="test",
        fed_recon=args.fed_recon,
    )

    if args.fed_recon:
        model_dataset_str = "fed-recon-" + args.dataset
    else:
        model_dataset_str = args.dataset

    model = PrototypicalNetwork(
        in_channels,
        args.embedding_size,
        hidden_size=args.hidden_size,
        pooling=args.pooling,
        backbone=args.backbone,
        l2_normalize_embeddings=args.l2_normalize_embeddings,
        drop_rate=args.drop_rate,
    )
    model.to(device=args.device)
    model.train()
    initial_lr = 1e-3
    optimizer = torch.optim.Adam(
        model.parameters(), lr=initial_lr, weight_decay=weight_decay
    )
    if args.cosine_lr:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.num_batches
        )
    else:
        # step decay following original paper: "cut the learning rate in half every 2000 episodes"
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2000, gamma=0.5)

    # Training loop
    test_set_accuracies = []
    losses = []
    all_meta_splits = []
    meta_batch_n = []
    if meta_val_dataset is not None:
        early_stopper = EarlyStopper(patience=20)
    for i in range(args.num_batches):
        model.zero_grad()

        batch = dataset.sample_meta_batch(
            batch_size=args.batch_size, sample_k_value=sample_k_value
        )
        if sample_k_value:
            _, n_classes_per_task_k_shots_per_class, _, _, _ = batch["train"][0].shape
            assert n_classes_per_task_k_shots_per_class % dataset.k_shots_per_class == 0
            k_shots_per_class = int(
                n_classes_per_task_k_shots_per_class / dataset.k_shots_per_class
            )
        else:
            k_shots_per_class = dataset.k_shots_per_class

        train_inputs, train_targets = batch["train"]
        train_inputs = train_inputs.to(device=args.device)
        train_embeddings = model(train_inputs)

        test_inputs, test_targets = batch["test"]
        test_inputs = test_inputs.to(device=args.device)
        test_targets = test_targets.to(device=args.device)
        test_embeddings = model(test_inputs)
        if not args.jsd and not args.mahala:
            prototypes = get_prototypes(
                train_embeddings,
                dataset.n_classes_per_task,
                k_shots_per_class,
                prototype_normal_std_noise=prototype_normal_std_noise,
            )  # -> [b, n, features]
            loss = prototypical_loss(
                prototypes,
                test_embeddings,
                test_targets,
                sum_loss_over_examples=sample_k_value,
            )
            loss_terms = None
        else:
            prototypes, loss, loss_terms = jsd_loss(
                train_embeddings,
                test_embeddings,
                test_targets,
                dataset.n_classes_per_task,
                dataset.k_shots_per_class,
                test_shots_per_class=test_shots,
                mahala=args.mahala,
                generalized_jsd_over_supports=args.generalized_jsd_supports,
            )
        loss.backward()
        optimizer.step()
        if args.cosine_lr:
            lr_scheduler.step()

        if i % eval_interval == 0:
            with torch.no_grad():
                accuracy = get_protonet_accuracy(
                    prototypes,
                    test_embeddings,
                    test_targets,
                    jsd=args.jsd,
                    mahala=args.mahala,
                )
                print("Loss at iteration {}: {}".format(i, loss.item()))
                print(
                    "Meta-train test-set accuracy at iteration {}: {}".format(
                        i, accuracy.item()
                    )
                )
                all_meta_splits.append("meta-train")
                meta_batch_n.append(i)
                test_set_accuracies.append(accuracy.item())

                if loss_terms is not None:
                    losses.append(loss_terms)
                else:
                    losses.append({"prototypical_loss": loss})

                if meta_val_dataset is not None:
                    val_acc, val_loss = eval_meta_val_set(
                        model, meta_val_dataset, args=args
                    )
                    print(
                        "Meta-val test-set accuracy at iteration {}: {}".format(
                            i, val_acc.item()
                        )
                    )
                    all_meta_splits.append("meta-val")
                    test_set_accuracies.append(val_acc.item())
                    meta_batch_n.append(i)

                    continue_training, best_val = early_stopper.continue_training(
                        val_acc.item(), i
                    )

                    if best_val:  # save model
                        if args.output_folder is not None:
                            if not os.path.exists(args.output_folder):
                                os.makedirs(args.output_folder)
                            filename = os.path.join(
                                args.output_folder,
                                f"protonet_{model_dataset_str}_{args.num_shots}-shot_{args.train_num_ways}-train-way.pt",
                            )
                            print(f"Saving best seen model to {filename}")
                            with open(filename, "wb") as f:
                                state_dict = model.state_dict()
                                torch.save(state_dict, f)

                    if not continue_training:
                        break

    # Save model
    if meta_val_dataset is None:
        if args.output_folder is not None:
            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder)
            filename = os.path.join(
                args.output_folder,
                f"protonet_{model_dataset_str}_{args.num_shots}-shot_{args.train_num_ways}-train-way.pt",
            )
            with open(filename, "wb") as f:
                state_dict = model.state_dict()
                torch.save(state_dict, f)

    mean_acc, ci = eval_meta_test_dataset(model, meta_test_dataset, args)

    if args.output_folder is not None:
        # Save training eval progress:
        df = pd.DataFrame(
            {
                "test-set accuracy": test_set_accuracies,
                "dataset": all_meta_splits,
                "meta-batch": meta_batch_n,
            }
        )
        df.to_csv(
            os.path.join(args.output_folder, "training_results.csv"),
            header=True,
            index=False,
        )

        losses = [(key, val) for x in losses for key, val in x.items()]
        loss_names, loss_values = map(list, zip(*losses))
        steps = list(range(len(loss_names)))
        pd.DataFrame(
            {
                "loss_name": loss_names,
                "loss_value": loss_values,
                "step": steps,
            }
        ).to_csv(
            os.path.join(args.output_folder, "training_losses.csv"),
            header=True,
            index=False,
        )

        # Save meta-test results
        with open(os.path.join(args.output_folder, "meta-test_results.txt"), "w") as f:
            f.write("{}+-{}%".format(mean_acc, ci))

        print("Results written to {}".format(args.output_folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prototypical Networks")

    parser.add_argument(
        "--data-dir", type=str, help="Path to omniglot or mini-imagenet."
    )
    parser.add_argument(
        "--dataset", type=str, help=f"{SUPPORTED_DATASETS}", default="omniglot"
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=5,
        help='Number of examples per class (k in "k-shot", default: 5).',
    )
    parser.add_argument(
        "--num-ways",
        type=int,
        default=5,
        help='Number of classes per task (N in "N-way", default: 5).',
    )
    parser.add_argument(
        "--train-num-ways",
        type=int,
        default=None,
        help='Number of classes per task (N in "N-way", default: 5) to use during training if different than test time.',
    )

    parser.add_argument("--backbone", type=str, default="4conv")

    parser.add_argument(
        "--embedding-size",
        type=int,
        default=64,
        help="Number of channels in the embedding/latent space (default: 64).",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Number of channels for each convolutional layer (default: 64).",
    )
    parser.add_argument(
        "--pooling",
        default=None,
        help=f"type of pooling layer to apply to feature vectors. Options are {SUPPORTED_POOLING_LAYERS}. Defaults to None, which will induce flattening of final feature maps following original prototypical networks paper.",
    )
    parser.add_argument(
        "--jsd",
        action="store_true",
        help="If provided, train with Jensen Shannon divergence.",
    )
    parser.add_argument(
        "--generalized-jsd-supports",
        action="store_true",
        help="If provided, train with Jensen Shannon divergence maximization over support distributions.",
    )
    parser.add_argument(
        "--mahala",
        action="store_true",
        help="If provided, train with prototypical network loss but with Mahalanobis loss instead of JSD between query and support.",
    )
    parser.add_argument("--l2-weight-decay", action="store_true")
    parser.add_argument("--drop-rate", type=float, default=None)
    parser.add_argument("--l2-normalize-embeddings", action="store_true")
    parser.add_argument("--prototype-normal-std-noise", default=None, type=float)
    parser.add_argument("--sample-k-value", default=False, action="store_true")

    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Path to the output folder for saving the model (optional).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of tasks in a mini-batch of tasks (default: 16).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50000,
        help="Number of batches the prototypical network is trained over (default: 50000).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers for data loading (default: 1).",
    )
    parser.add_argument(
        "--use-cuda", action="store_true", help="Use CUDA if available."
    )
    parser.add_argument(
        "--cosine-lr", action="store_true", help="Cosine anneal the learning rate."
    )
    parser.add_argument(
        "--n-val-during-training",
        type=int,
        default=100,
        help="Number of val-set evaluations during training.",
    )
    parser.add_argument(
        "--fed-recon",
        action="store_true",
        help="Use the federated reconnaissance splits.",
    )

    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )

    train(args)
