"""
Evaluates a model on the federated reconnaissance benchmark.

To use this API, implement the following required methods onto a model class:
    model.update_base_memory(base_train_dataset: torch.Dataset)
    model.test_model(test_dataset: torch.Dataset)
    model.eval_model(eval_data: torch.Tensor, eval_labels: torch.Tensor)
    model.merge_models(new_models: Dict[int, "MyModelClass"], new_classes: Dict[int, List[int]]

Optionally, to have the script train your model from scratch, you can also implement:
    model.train_base_model(base_train_dataset: torch.Dataset, config: Dict[str, Any])
    # where config is an optional dictionary specifying hyperparameters for training the model

This benchmark runs in 2 main modes, one in which examples are sampled with replacement in the field and one in which examples are not sampled with replacement.
To force each example to only be seen once, add "sample_without_replacement": true, to the config.json file.

Finally, add a function to build your model class to the MODEL_BUILDERS dictionary
"""
import os
import sys
from typing import Dict, Any

import torch

# This is needed to make script work even when inside package and we can't change the python path variable (such as in a singularity exec call):

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mtm.fed_recon.client_server import Server
from mtm.fed_recon.mini_imagenet import MiniImagenet
from mtm.models.protonet.model import FederatedProtoNet
from mtm.models.gradient_based.icarl import ICaRL
from mtm.models.gradient_based.upper import Upper
from mtm.models.gradient_based.lower import Lower
from mtm.util.experiment import ExperimentLogger
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="configuration file")
parser.add_argument("--use-cuda", action="store_true", help="Use CUDA if available.")
parser.add_argument(
    "--output_path", help="Path to directory to write results in.", default=None
)

N_CLIENTS = 5
N_CLASSES_TO_SAMPLE_PER_MISSION = 5
MAX_MISSIONS = 100

# Hyperparams to remove the effects of federated learning:
# N_CLIENTS = 1
# N_CLASSES_TO_SAMPLE_PER_MISSION = 50
# MAX_MISSIONS = 1
# TRAINING_EXAMPLES_PER_CLASS = 500  # 1


def build_protonet(config: Dict[str, Any], use_cuda: bool = True) -> FederatedProtoNet:
    model_path = config["model_path"]
    store_all_embeddings_in_memory = config["store_all_embeddings_in_memory"]

    in_channels = 3

    try:
        hidden_size = config["hidden_size"]
    except KeyError:
        hidden_size = 64
    try:
        embedding_size = config["embedding_size"]
    except KeyError:
        embedding_size = 64
    try:
        backbone = config["backbone"]
    except KeyError:
        backbone = "4conv"
    try:
        pooling = config["pooling"]
    except KeyError:
        pooling = None
    try:
        distance_function = config["distance_function"]
    except KeyError:
        distance_function = "euclidean"

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    model = FederatedProtoNet(
        in_channels=in_channels,
        out_channels=embedding_size,
        hidden_size=hidden_size,
        distance_function=distance_function,
        store_all_embeddings_in_memory=store_all_embeddings_in_memory,
        pooling=pooling,
        backbone=backbone,
        device=device,
    )
    if use_cuda and torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))  # FIXME explicitly map to gpu
        model = model.to(device)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model


def build_IcaRL(config) -> ICaRL:
    n_classes = 50
    json_file_icarl = config["icarl_json_file"]
    model_config = json.load(open(json_file_icarl, "r"))
    model = ICaRL(n_classes, model_config)
    return model


def build_Upper(config) -> Upper:
    n_classes = 50
    upper_json_file = config["upper_json_file"]
    model_config = json.load(open(upper_json_file, "r"))
    model = Upper(n_classes, model_config)
    return model


def build_Lower(config) -> Lower:
    n_classes = 50
    lower_json_file = config["lower_json_file"]
    model_config = json.load(open(lower_json_file, "r"))
    model = Lower(n_classes, model_config)
    return model


MODEL_BUILDERS = {
    "federated_protonet": build_protonet,
    "IcaRL": build_IcaRL,
    "Upper": build_Upper,
    "Lower": build_Lower,
}


def done(
    mission: int,
    environment: MiniImagenet,
    sample_without_replacement: bool,
    max_missions: int,
):
    if sample_without_replacement:
        _done = (
            len(environment.previously_sampled_field_train_images)
            >= environment.n_total_field_training_examples
        )
    else:
        _done = mission >= max_missions
    return _done


def main(args):
    config = json.load(open(args.config, "r"))
    images_path = config["images_path"]

    if args.output_path is None:
        output_path = config["output_path"]
    else:  # Override config
        output_path = args.output_path
    model_type = config["model_type"]
    sample_wo_replacement = config["sample_without_replacement"]
    training_examples_per_class = config["training_examples_per_class"]
    train_base_model = config["train_base_model"]

    build_model = MODEL_BUILDERS[model_type]

    model = build_model(config)

    environment = MiniImagenet(
        images_path,
        training_examples_per_class_per_mission=training_examples_per_class,
        n_classes_per_mission=N_CLASSES_TO_SAMPLE_PER_MISSION,
        sample_without_replacement=sample_wo_replacement,
    )

    experiment = ExperimentLogger(output_path)
    server = Server(
        N_CLIENTS,
        environment,
        model,
        train_base_model=train_base_model,
        update_base_memory=True,
        experiment=experiment,
    )

    i = 0
    while not done(
        mission=i,
        environment=environment,
        sample_without_replacement=sample_wo_replacement,
        max_missions=MAX_MISSIONS,
    ):
        print(
            f"Mission {i} with {server.num_clients}: {len(server.reconned_classes)} of {len(environment.field_classes)} classes learned"
        )
        server.send_mission(mission_number=i)
        i += 1
    server.save_results()


if __name__ == "__main__":
    arguments = parser.parse_args()
    main(arguments)
