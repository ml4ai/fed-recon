import os
import sys

# to make it run on hpc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from mtm.fed_recon.mini_imagenet import MiniImagenet
from mtm.models.gradient_based.icarl import ICaRL
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="configuration file")


def main(args):
    config = json.load(open(args.config, "r"))
    images_path = config["images_path"]
    # with hold_out for tune hyper-parameters
    n_classes = 40
    # without hold_out for train base
    # n_classes = 50

    json_file_icarl = config["icarl_json_file"]
    model_config = json.load(open(json_file_icarl, "r"))
    model = ICaRL(n_classes, model_config)

    # for tune-hyper parameter
    hold_out = True
    base_val = True

    # for train base model
    # hold_out = False
    # base_val = True

    environment = MiniImagenet(
        images_path,
        training_examples_per_class_per_mission=30,
        n_classes_per_mission=5,
        sample_without_replacement=True,
        hold_out_bool=hold_out,
        base_val_bool=base_val,
    )

    # tune hyper parameter
    model.tune_hyperparameters(
        environment.base_train,
        environment.base_val,
        environment.hold_out_train,
        environment.hold_out_val,
    )

    # train base
    # model.train_model(environment.base_train, environment.base_val, save_model=True)


if __name__ == "__main__":
    arguments = parser.parse_args()
    main(arguments)
