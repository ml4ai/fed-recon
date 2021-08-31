"""
Runner script that evaluates prototypical network models on class incremental learning.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from mtm.single_client.eval_cil import SUPPORTED_DATASETS, eval_cl
from mtm.models.protonet.model import PrototypicalNetworkCL


def parse_args():
    parser = argparse.ArgumentParser("Prototypical Networks")

    parser.add_argument("--data-dir", type=str, help="Path to mini-imagenet.")
    parser.add_argument(
        "--dataset", type=str, help=f"{SUPPORTED_DATASETS}", default="omniglot"
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=30,
        help='Number of examples per class (k in "k-shot", default: 30).',
    )

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
        "--model-path",
        type=str,
        required=True,
        help="Path to the model to load.",
    )
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
        help="Number of tasks in a mini-batch of tasks (default: 1).",
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
        "--mahalanobis",
        action="store_true",
        help="Use Mahalanobis distance for evaluation.",
    )

    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )
    return args


def main(args):
    print(f"torch version:")
    print(torch.__version__)
    if args.dataset == "omniglot":
        in_channels = 1
    elif args.dataset == "mini-imagenet":
        in_channels = 3
    else:
        raise NotImplementedError(
            f"--dataset should be in {SUPPORTED_DATASETS} but {args.dataset} was provided"
        )
    if args.mahalanobis:
        distance_function = "mahalanobis"
    else:
        distance_function = "euclidean"
    model = PrototypicalNetworkCL(
        in_channels=in_channels,
        out_channels=args.embedding_size,
        hidden_size=args.hidden_size,
        distance_function=distance_function,
    )

    # FIXME: need to implement cuda support
    if not args.use_cuda:
        map_location = "cpu"
    else:
        map_location = None
    # model.to(args.device)
    # model.load_state_dict(torch.load(args.model_path, map_location=map_location))

    model.load_state_dict(torch.load(args.model_path, map_location=map_location))

    model.eval()

    print(f"device in main {args.device}")
    all_accs, all_n_classes = eval_cl(
        model, root_dir=args.data_dir, dataset=args.dataset, device=args.device
    )

    df = pd.DataFrame({"accuracy": all_accs, "n_classes": all_n_classes})

    print("Summary of class-incremental learning results:")
    grouped = df.groupby("n_classes")
    print(grouped["accuracy"].agg([np.mean, np.std]))

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    results_path = os.path.join(
        args.output_folder, f"{args.dataset}-cl-meta-test-results.csv"
    )
    df.to_csv(
        results_path,
        header=True,
        index=False,
    )
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
