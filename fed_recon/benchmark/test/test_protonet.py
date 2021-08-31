import torch
from fed_recon.benchmark.client_server import Server
from fed_recon.benchmark.mini_imagenet import MiniImagenet
from fed_recon.models.protonet.model import FederatedProtoNet
from fed_recon.util.experiment import ExperimentLogger

num_clients = 2
images_path = "data/mini-imagenet/mini-imagenet/images"
model_path = "data/experiment_results/mini-imagenet/protonet_miniimagenet_5-way_5-shot_step-decay-lr_1605127083/protonet_omniglot_5-shot_5-way.pt"
output_path = "/tmp/fedrecon"
environment = MiniImagenet(images_path, training_examples_per_class_per_mission=5)
n_classes = len(environment.base_labels)

in_channels = 3
hidden_size = 64
embedding_size = 64
distance_function = "euclidean"


def test():
    model = FederatedProtoNet(
        in_channels=in_channels,
        out_channels=embedding_size,
        hidden_size=hidden_size,
        distance_function=distance_function,
        store_all_embeddings_in_memory=False,
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    experiment = ExperimentLogger(output_path)
    server = Server(
        num_clients,
        environment,
        model,
        train_base_model=False,
        update_base_memory=True,
        experiment=experiment,
    )

    for i in range(2):
        server.send_mission(mission_number=i)

    server.save_results()


def test_store_all_embeddings():
    model = FederatedProtoNet(
        in_channels=in_channels,
        out_channels=embedding_size,
        hidden_size=hidden_size,
        distance_function=distance_function,
        store_all_embeddings_in_memory=True,
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    server = Server(
        num_clients, environment, model, train_base_model=False, update_base_memory=True
    )

    # dataset, class_list = environment.get_mission_train_data()
    for i in range(1):
        server.send_mission(i)


if __name__ == "__main__":
    test()
    # test_store_all_embeddings()
