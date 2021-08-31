"""
Prototypical networks for Federated Reconnaissance.

This file currently contains 3 model definitions:
    multi-client protonet for fed recon
    single client protonet for CL
    vanilla prototypical network

References:
    [1] Snell et al. 2017. Prototypical Networks for Few-shot Learning. https://arxiv.org/abs/1703.05175
    [2] Prototypical network model adapted from: https://github.com/tristandeleu/pytorch-meta and
    [3] https://github.com/jakesnell/prototypical-networks
"""
from typing import Dict, List, Any, Optional, Union

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision
from tqdm import tqdm

from fed_recon.benchmark.mini_imagenet import BaseTrain
from fed_recon.util.metrics import get_accuracy
from fed_recon.util.util import GeM
from fed_recon.util.viz_util import plot_image_batch


SUPPORTED_POOLING_LAYERS = {"average", "Gem"}


def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


def build_4conv_protonet_encoder(
    in_channels: int,
    hidden_size: int,
    out_channels: int,
    drop_rate: Optional[float] = None,
) -> nn.Sequential:
    layers = [
        conv3x3(in_channels, hidden_size),
        conv3x3(hidden_size, hidden_size),
        conv3x3(hidden_size, hidden_size),
        conv3x3(hidden_size, out_channels),
    ]
    if drop_rate is not None and drop_rate != 0.0:
        layers.append(nn.Dropout(p=drop_rate))
    return nn.Sequential(*layers)


def build_resnet18_encoder(drop_rate: Optional[float] = None):
    encoder = torchvision.models.resnet18(pretrained=False)
    encoder = list(encoder.children())[:-2]
    if drop_rate is not None and drop_rate != 0.0:
        encoder.append(nn.Dropout(p=drop_rate))
    encoder = nn.Sequential(*encoder)
    return encoder


class FederatedProtoNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_size=64,
        distance_function: str = "euclidean",
        store_all_embeddings_in_memory: bool = False,
        pooling: Optional[str] = None,
        backbone: str = "4conv",
        device: Union[str, torch.device] = "cpu",
    ):
        """Multi-client, continual learning prototypical network for federated reconnaissance."""

        super().__init__()
        self.supported_distance_functions = {"euclidean", "mahalanobis"}
        assert distance_function.lower() in self.supported_distance_functions
        self.supported_backbones = {"4conv", "resnet18"}
        assert backbone in self.supported_backbones
        self.distance_function: str = distance_function.lower()

        if pooling is not None:
            assert pooling in SUPPORTED_POOLING_LAYERS
        self.pooling = pooling

        self.store_all_embeddings_in_memory = store_all_embeddings_in_memory

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        if backbone == "resnet18":
            self.encoder = build_resnet18_encoder()
        elif backbone == "4conv":
            self.encoder = build_4conv_protonet_encoder(
                in_channels, hidden_size, out_channels
            )
        else:
            raise ValueError(
                f"Unsupported backbone {backbone} not in {self.supported_backbones}"
            )
        if pooling == "Gem":
            self.gem_pooling = GeM()
        else:
            self.gem_pooling = None

        self.prototypes: Dict[
            int, torch.FloatTensor
        ] = {}  # Map from class index to prototype
        self.examples_per_class: Dict[
            int, int
        ] = {}  # Map from class index to n examples seen thus far
        self.index_label_map = {}  # Map from index to label
        self.label_index_map = {}  # Map from label to index

        self.embeddings: Dict[
            int, torch.Tensor
        ] = {}  # Map from class index to embeddings

    def forward(self, inputs):
        # inputs should be shape [b, k * n, ch, rows, cols]
        assert len(inputs.shape) == 5
        inputs_reshaped = inputs.view(
            -1, *inputs.shape[2:]
        )  # -> [b * k * n, ch, rows, cols]
        embeddings = self.encoder(inputs_reshaped)  # -> [n * k, ch, rows, cols]
        if self.pooling is None:
            embeddings_reshaped = embeddings.view(
                *inputs.shape[:2], -1
            )  # -> [n * k, ch * rows * cols]
        elif self.pooling == "average":
            embeddings_reshaped = embeddings.mean(dim=[-1, -2])
        elif self.pooling == "Gem":
            embeddings_reshaped = self.gem_pooling(embeddings)
        else:
            raise ValueError
        return embeddings_reshaped

    def train_base_model(self, base_train_dataset: Dataset, config: Dict[str, Any]):
        """
        :param base_train_dataset: train the model with base_train_dataset
        :param config: dictionary of hyperparameters for meta pretraining
        :return: None
        """
        raise NotImplementedError(f"todo: implement training here")

    # FIXME: implement these methods so that we can merge the 3 model classes into 1
    # def evaluate(
    #     self,
    #     train_inputs,
    #     train_labels,
    #     test_inputs,
    #     test_labels,
    #     n_train_classes: int,
    #     k_shots: int,
    # ) -> torch.FloatTensor:
    #     """Computes the mean accuracy of predictions on test_inputs."""
    #     train_embeddings = self.forward(train_inputs)
    #     test_embeddings = self.forward(test_inputs)
    #     all_class_prototypes = self._get_prototypes(train_embeddings, train_labels, n_train_classes, k_shots)
    #     if self.distance_function == "euclidean":
    #         mahala = False
    #     elif self.distance_function == "mahalanobis":
    #         mahala = True
    #     else:
    #         raise NotImplementedError(f"Distance function must be in {self.supported_distance_functions} but is {self.distance_function}")
    #
    #     acc = get_protonet_accuracy(all_class_prototypes, test_embeddings, test_labels, mahala=mahala)
    #     return acc
    #
    # def _get_prototypes(self, train_embeddings: torch.Tensor, train_labels: torch.Tensor, n_train_classes: int, k_shots: int):
    #     new_prototypes = get_prototypes(
    #         train_embeddings, n_train_classes, k_shots
    #     )  # -> [b, n, features]
    #
    #     # Add each prototype to the model:
    #     assert len(new_prototypes.shape) == 3
    #     assert new_prototypes.shape[0] == 1
    #     new_prototypes = new_prototypes[
    #         0
    #     ]  # Assume single element in batch dimension. Now [n_train_classes, features]
    #     class_indices = np.unique(train_labels.numpy())
    #     # class_indices = torch.unique(train_labels)
    #     for i, cls_index in enumerate(class_indices):
    #         self.update_prototype_for_class(
    #             cls_index, new_prototypes[i, :], train_labels.shape[1]
    #         )
    #     all_class_prototypes = [
    #         self.prototypes[key] for key in sorted(self.prototypes.keys())
    #     ]
    #     all_class_prototypes: torch.FloatTensor = torch.stack(
    #         all_class_prototypes, dim=0
    #     ).unsqueeze(
    #         0
    #     )  # -> [1, n, features]
    #     return all_class_prototypes

    @property
    def n_classes(self):
        assert len(self.label_index_map) == len(
            self.index_label_map
        ), f"mappings from class labels to indexes corrupted {self.label_index_map} {self.index_label_map}"
        return len(self.label_index_map)

    def add_class(self, new_class: int):
        """Adds a new class's info"""
        assert isinstance(new_class, int)
        if new_class not in self.label_index_map:
            i = self.n_classes
            self.label_index_map[new_class] = i
            self.index_label_map[i] = new_class

    def update_base_memory(self, base_train_dataset: BaseTrain):
        self.eval()
        if self.store_all_embeddings_in_memory:
            return self._update_base_memory_embeddings(base_train_dataset)
        batch_size = 500
        total_training_examples_per_class = base_train_dataset.total_examples_per_class
        assert (
            total_training_examples_per_class % batch_size == 0
        ), "Implementation requires that the loader returns batches from a SINGLE class"

        with torch.no_grad():
            train_loader = DataLoader(
                base_train_dataset, batch_size=batch_size, shuffle=False
            )
            i = -1
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                assert (
                    len(labels.unique()) == 1
                ), "Implementation requires that the loader returns batches from a SINGLE class"
                if int(labels[0]) not in self.label_index_map:
                    i += 1
                    # if i == 2:  # uncomment for quicker debugging
                    #     break
                    self.label_index_map[int(labels[0])] = i

                embeddings = self.forward(inputs.unsqueeze(0))
                # prototype.shape
                # torch.Size([1, batch_size, 1600])

                prototype = get_prototypes(
                    embeddings, n_classes=1, k_shots_per_class=batch_size
                )

                # Add each prototype to the model:
                assert len(prototype.shape) == 3
                assert prototype.shape[0] == 1
                prototype = prototype[
                    0
                ]  # Assume single element in batch dimension. Now [n_train_classes, features]
                self.update_prototype_for_class(i, prototype[0, :], labels.shape[0])

        # Swap keys with vals
        self.index_label_map = dict((v, k) for k, v in self.label_index_map.items())

    def _update_base_memory_embeddings(self, base_train_dataset: BaseTrain):
        batch_size = 50
        total_training_examples_per_class = base_train_dataset.total_examples_per_class
        assert (
            total_training_examples_per_class % batch_size == 0
        ), "Implementation requires that the loader returns batches from a SINGLE class"

        with torch.no_grad():
            train_loader = DataLoader(
                base_train_dataset, batch_size=batch_size, shuffle=False
            )
            i = -1
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                assert (
                    len(labels.unique()) == 1
                ), "Implementation requires that the loader returns batches from a SINGLE class"

                if int(labels[0]) not in self.label_index_map:
                    i += 1
                    # if i == 2:  # uncomment for quicker debugging
                    #     break
                    self.label_index_map[int(labels[0])] = i

                embeddings = self.forward(inputs.unsqueeze(0))
                # torch.Size([1, batch_size, 1600])
                # Add each prototype to the model:
                assert len(embeddings.shape) == 3
                assert embeddings.shape[0] == 1
                embeddings = embeddings[
                    0
                ]  # Assume single element in batch dimension. Now [n_train_classes, features]
                self._update_embeddings_for_class(i, embeddings)

        # Swap keys with vals
        self.index_label_map = dict((v, k) for k, v in self.label_index_map.items())

    def _update_embeddings(
        self,
        mission_data: torch.Tensor,
        mission_labels: torch.Tensor,
        mission_classes: List,
    ):
        """
        :param mission_data: mission data [will be dataset later, tensor for now]
        :param mission_labels: mission_labels
        :param mission_classes: new classes learned from mission
        :return: None
        """
        for new_class in mission_classes:
            self.add_class(new_class)

        new_data = {}  # Map from class label to features
        for inputs, label in zip(mission_data, mission_labels):
            try:
                new_data[int(label.item())].append(inputs)
            except KeyError:
                new_data[int(label.item())] = [inputs]

        for label, inputs in new_data.items():
            inputs = torch.stack(inputs, dim=0)
            inputs = inputs.to(self.device)
            embeddings = self.forward(inputs.unsqueeze(0))[
                0
            ]  # single element in batch dim
            # embeddings.shape
            # torch.Size([1, k_shots, 1600])
            self._update_embeddings_for_class(self.label_index_map[label], embeddings)

    def update_model(
        self,
        mission_data: torch.Tensor,
        mission_labels: torch.Tensor,
        mission_classes: List,
    ):
        """
        :param mission_data: mission data [will be dataset later, tensor for now]
        :param mission_labels: mission_labels
        :param mission_classes: new classes learned from mission
        :return: None
        """
        if self.store_all_embeddings_in_memory:
            return self._update_embeddings(
                mission_data, mission_labels, mission_classes
            )
        for new_class in mission_classes:
            self.add_class(new_class)

        new_data = {}  # Map from class label to features
        for inputs, label in zip(mission_data, mission_labels):
            try:
                new_data[int(label.item())].append(inputs)
            except KeyError:
                new_data[int(label.item())] = [inputs]

        for label, inputs in new_data.items():
            k_shots = len(inputs)
            inputs = torch.stack(inputs, dim=0)
            inputs = inputs.to(self.device)
            embeddings = self.forward(inputs.unsqueeze(0))
            # embeddings.shape
            # torch.Size([1, k_shots, 1600])

            prototype = get_prototypes(
                embeddings, n_classes=1, k_shots_per_class=k_shots
            )

            # Add each prototype to the model:
            assert len(prototype.shape) == 3
            assert prototype.shape[0] == 1
            prototype = prototype[
                0
            ]  # Assume single element in first dimension. Now [n_train_classes, features]
            self.update_prototype_for_class(
                self.label_index_map[label], prototype[0, :], k_shots
            )

    def eval_model(self, eval_data: torch.Tensor, eval_labels: torch.Tensor):
        """
        :param eval_data: tensor for now, will be a dataset later
        :param eval_labels: tensor for now, will be a dataset later
        :return:
        """
        assert isinstance(eval_data, torch.Tensor)
        self.eval()

        if self.distance_function == "euclidean":
            mahala = False
        elif self.distance_function == "mahalanobis":
            mahala = True

        with torch.no_grad():
            #  get the prototypes for all classes
            if not self.store_all_embeddings_in_memory:
                all_class_prototypes = [
                    self.prototypes[key] for key in sorted(self.prototypes.keys())
                ]
                all_class_prototypes: torch.FloatTensor = torch.stack(
                    all_class_prototypes, dim=0
                ).unsqueeze(
                    0
                )  # -> [1, n, features]
            else:
                all_class_prototypes = [
                    torch.mean(self.embeddings[key], dim=0)
                    for key in sorted(self.embeddings.keys())
                ]
                all_class_prototypes: torch.FloatTensor = torch.stack(
                    all_class_prototypes, dim=0
                ).unsqueeze(
                    0
                )  # -> [1, n, features]

            eval_data, eval_labels = eval_data.to(self.device), eval_labels.to(
                self.device
            )

            eval_dataset = torch.utils.data.TensorDataset(eval_data, eval_labels)
            eval_loader = DataLoader(eval_dataset, batch_size=10)

            accs = []
            for images, labels in eval_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                embeddings = self.forward(images.unsqueeze(0))
                acc = self.get_protonet_accuracy(
                    all_class_prototypes,
                    embeddings,
                    labels.unsqueeze(0),
                    mahala=mahala,
                )
                accs.append(float(acc))
            acc = np.mean(accs)
        return acc

    def get_protonet_accuracy(
        self,
        prototypes: torch.FloatTensor,
        embeddings: torch.FloatTensor,
        targets: torch.Tensor,
        jsd: bool = False,
        mahala: bool = False,
    ) -> torch.FloatTensor:
        """Compute the accuracy of the prototypical network on the test/query points.

        Parameters
        ----------
        prototypes : `torch.FloatTensor` instance
            A tensor containing the prototypes for each class. This tensor has shape
            `(meta_batch_size, num_classes, embedding_size)`.
        embeddings : `torch.FloatTensor` instance
            A tensor containing the embeddings of the query points. This tensor has
            shape `(meta_batch_size, num_examples, embedding_size)`.
        targets : `torch.LongTensor` instance
            A tensor containing the targets of the query points. This tensor has
            shape `(meta_batch_size, num_examples)`.
        jsd: bool. If True, only use first half of prototypes and embeddings.


        Returns
        -------
        accuracy : `torch.FloatTensor` instance
            Mean accuracy on the query points.
        """
        if mahala or jsd:
            embedding_dim = prototypes.shape[-1]
            prototypes, sd = (
                prototypes[:, :, 0 : embedding_dim // 2],
                prototypes[:, :, embedding_dim // 2 :],
            )
            sd = sd.mul(0.5).exp_()
            embeddings = embeddings[:, :, 0 : embedding_dim // 2]
            sq_distances = torch.sum(
                ((prototypes.unsqueeze(1) - embeddings.unsqueeze(2)) ** 2) / sd, dim=-1
            )
        else:
            sq_distances = torch.sum(
                (prototypes.unsqueeze(1) - embeddings.unsqueeze(2)) ** 2, dim=-1
            )
        _, predictions = torch.min(sq_distances, dim=-1)
        batch = predictions.shape[0]
        items = predictions.shape[1]
        for i in range(batch):
            for j in range(items):
                predictions[i, j] = self.index_label_map[int(predictions[i, j])]
        return get_accuracy(predictions, targets)

    def test_model(self, test_dataset: Dataset):
        assert isinstance(test_dataset, Dataset)
        self.eval()

        if self.distance_function == "euclidean":
            mahala = False
        elif self.distance_function == "mahalanobis":
            mahala = True

        #  get the prototypes for all classes
        if not self.store_all_embeddings_in_memory:
            all_class_prototypes = [
                self.prototypes[key] for key in sorted(self.prototypes.keys())
            ]
            all_class_prototypes: torch.FloatTensor = torch.stack(
                all_class_prototypes, dim=0
            ).unsqueeze(
                0
            )  # -> [1, n, features]
        else:
            all_class_prototypes: List[torch.Tensor] = [
                torch.mean(self.embeddings[key], dim=0)
                for key in sorted(self.embeddings.keys())
            ]
            all_class_prototypes: torch.FloatTensor = torch.stack(
                all_class_prototypes, dim=0
            ).unsqueeze(
                0
            )  # -> [1, n, features]

        test_loader = DataLoader(
            test_dataset, batch_size=300, shuffle=False, pin_memory=True
        )  # , num_workers=8)
        accs = []
        with torch.no_grad():
            for i, (data, test_label) in enumerate(test_loader):
                # if i == 2:  # uncomment for quicker debugging
                #     break
                # plot_image_batch(data, test_label)

                # Embed the query points
                data, test_label = data.to(self.device), test_label.to(self.device)
                test_embeddings = self.forward(data.unsqueeze(0))  # -> [1, b, 1600]
                acc_i = self.get_protonet_accuracy(
                    all_class_prototypes, test_embeddings, test_label, mahala=mahala
                )
                accs.append(float(acc_i))

        acc = np.mean(accs)
        return acc

    def update_prototype_for_class(
        self, class_index: int, proto: torch.FloatTensor, n_samples: int
    ):
        """
        Computes unbiased, online average prototypes for each class_index.
        if self.distance_function = "mahalanobis", it is assumed that the second half of the prototype vector is the predicted standard deviation

        proto: is a vector representing the centroid of n_samples vectors

        References:
            [1] West, D.H.D., 1979. Updating mean and variance estimates: An improved method. Communications of the ACM, 22(9), pp.532-535.
            [2] Schubert, E. and Gertz, M., 2018, July. Numerically stable parallel computation of (co-) variance. In Proceedings of the 30th International Conference on Scientific and Statistical Database Management (pp. 1-12).

        Note:
            Putting the update math or inputs into log space didn't improve numerical stability
        """
        #  TODO: train prototypical networks that are robust to small floating point operation errors by adding noise to embeddings/prototypes
        assert isinstance(class_index, int)
        assert len(proto.shape) == 1
        proto = proto.detach().clone()
        with torch.no_grad():
            try:
                # Numerically stable version following [1] and [2]:
                old_proto = self.prototypes[class_index]
                self.examples_per_class[class_index] += n_samples
                self.prototypes[class_index] = old_proto + (
                    (n_samples / self.examples_per_class[class_index])
                    * (proto - old_proto)
                )

                # Implementation following paper:
                # old_proto = self.prototypes[class_index]
                # old_k = self.examples_per_class[class_index]
                # self.examples_per_class[class_index] += n_samples
                # new_k = self.examples_per_class[class_index]
                # self.prototypes[class_index] = ((old_k * old_proto) + (n_samples * proto)) / new_k

            except KeyError:
                self.prototypes[class_index] = proto
                self.examples_per_class[class_index] = n_samples

        # print("DynamicProtoNet.examples_per_class", self.examples_per_class)

    def _update_embeddings_for_class(self, class_index: int, embeddings: torch.Tensor):
        """
        Stores all embeddings seen thus for class `class_index`
        More numerically stable, though higher memory consumption and communication overhead than using update_prototype_for_class
        """
        assert self.store_all_embeddings_in_memory
        assert isinstance(class_index, int)
        assert (
            len(embeddings.shape) == 2
        )  # Should be tensor of shape [n_examples_for_class_index, embedding_dim]
        # embeddings = embeddings.detach().clone()
        embeddings = embeddings.detach()
        with torch.no_grad():
            try:
                old_embeddings = self.embeddings[class_index]
                self.embeddings[class_index] = torch.cat(
                    [old_embeddings, embeddings], dim=0
                )
            except KeyError:
                self.embeddings[class_index] = embeddings

        print(
            f"class index {class_index} embeddings shape: {self.embeddings[class_index].shape}"
        )

    def _merge_models_stored_embeddings(
        self,
        new_models: Dict[int, "FederatedProtoNet"],
        new_classes: Dict[int, List[int]],
    ):
        """
        :param new_models: new models learned by clients from mission
        :param new_classes: new classes learned by clients from mission
        :return: None
        """
        assert self.store_all_embeddings_in_memory
        map_class_index_to_prev_emeddings = {
            class_index: self.embeddings[class_index].shape[0]
            for class_index in self.index_label_map.keys()
        }
        for client_id, other in new_models.items():
            for label in other.label_index_map.keys():
                if label not in new_classes[client_id]:
                    continue
                self.add_class(label)
                class_index = self.label_index_map[label]
                # We rely on the fact that each client will stack embeddings after previous embeddings provided by server
                client_embeddings = other.embeddings[other.label_index_map[label]]
                print(
                    f"client {client_id} embeddings for class index {class_index}: {client_embeddings.shape}"
                )
                if class_index in map_class_index_to_prev_emeddings:
                    new_client_embeddings = client_embeddings[
                        map_class_index_to_prev_emeddings[class_index] :
                    ]
                    print(
                        f"Previously class! New embeddings from client shape: {client_embeddings.shape}"
                    )
                else:
                    print(
                        f"New class! New embeddings from client shape: {client_embeddings.shape}"
                    )
                    new_client_embeddings = client_embeddings
                self._update_embeddings_for_class(class_index, new_client_embeddings)

    def merge_models(
        self,
        new_models: Dict[int, "FederatedProtoNet"],
        new_classes: Dict[int, List[int]],
    ):
        """
        :param new_models: new models learned by clients from mission
        :param new_classes: new classes learned by clients from mission
        :return: None
        """
        if self.store_all_embeddings_in_memory:
            return self._merge_models_stored_embeddings(new_models, new_classes)
        print("[Model] merging models")
        old_n_examples_per_class = {
            class_index: n_examples
            for class_index, n_examples in self.examples_per_class.items()
        }
        for client_id, other in new_models.items():
            for label in other.label_index_map.keys():
                if label not in new_classes[client_id]:
                    continue
                self.add_class(label)
                new_class_index = self.label_index_map[label]
                try:
                    old_n_examples = old_n_examples_per_class[new_class_index]
                except KeyError:
                    old_n_examples = 0
                new_n_examples = (
                    other.examples_per_class[other.label_index_map[label]]
                    - old_n_examples
                )
                self.update_prototype_for_class(
                    new_class_index,
                    other.prototypes[other.label_index_map[label]],
                    new_n_examples,
                )


class PrototypicalNetwork(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_size=64,
        pooling: Optional[str] = None,
        backbone: str = "4conv",
        l2_normalize_embeddings: bool = False,
        drop_rate: Optional[float] = None,
    ):
        """Standard prototypical network"""
        super().__init__()
        self.supported_backbones = {"4conv", "resnet18"}
        assert backbone in self.supported_backbones
        self.pooling = pooling
        self.backbone = backbone
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        if self.backbone == "resnet18":
            self.encoder = build_resnet18_encoder(drop_rate=drop_rate)
        elif self.backbone == "4conv":
            self.encoder = build_4conv_protonet_encoder(
                in_channels, hidden_size, out_channels, drop_rate=drop_rate
            )
        else:
            raise ValueError(
                f"Unsupported backbone {self.backbone} not in {self.supported_backbones}"
            )

        if self.pooling is not None:
            assert self.pooling in SUPPORTED_POOLING_LAYERS
        if self.pooling == "Gem":
            self.gem_pooling = GeM()
        else:
            self.gem_pooling = None

        self.l2_normalize_embeddings = l2_normalize_embeddings

    def forward(self, inputs):
        batch, nk, _, _, _ = inputs.shape
        inputs_reshaped = inputs.view(
            -1, *inputs.shape[2:]
        )  # -> [b * k * n, input_ch, rows, cols]

        embeddings = self.encoder(
            inputs_reshaped
        )  # -> [b * n * k, embedding_ch, rows, cols]
        # TODO: add optional support for half for further prototype/embedding compression
        # embeddings = embeddings.type(torch.float16)
        # RuntimeError: "clamp_min_cpu" not implemented for 'Half'
        if self.pooling is None:
            embeddings_reshaped = embeddings.view(
                *inputs.shape[:2], -1
            )  # -> [b, n * k, embedding_ch * rows * cols] (4608 for resnet 18)
        elif self.pooling == "average":
            embeddings_reshaped = embeddings.mean(dim=[-1, -2]).reshape(
                batch, nk, -1
            )  # -> [b, n * k, embedding_ch] (512 for resnet 18)
        elif self.pooling == "Gem":
            embeddings_reshaped = (
                self.gem_pooling(embeddings).squeeze(-1).squeeze(-1).unsqueeze(0)
            )
        else:
            raise ValueError
        if self.l2_normalize_embeddings:
            embeddings_reshaped = torch.nn.functional.normalize(
                embeddings_reshaped, p=2, dim=2
            )
        return embeddings_reshaped

    def evaluate(
        self,
        train_inputs: torch.Tensor,
        train_labels: torch.Tensor,
        test_inputs: torch.Tensor,
        test_labels: torch.Tensor,
        n_classes: int,
        k_shots: int,
    ) -> torch.FloatTensor:
        """Computes the mean accuracy of predictions on test_inputs."""
        train_embeddings = self.forward(train_inputs)
        test_embeddings = self.forward(test_inputs)
        prototypes = get_prototypes(
            train_embeddings, n_classes, k_shots
        )  # -> [b, n, features]
        acc = get_protonet_accuracy(prototypes, test_embeddings, test_labels)
        return acc


class PrototypicalNetworkCL(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_size=64,
        distance_function: str = "euclidean",
        backbone: str = "4conv",
    ):
        """Single client prototypical network for continual learning."""
        super().__init__()
        self.supported_distance_functions = {"euclidean", "mahalanobis"}
        assert distance_function.lower() in self.supported_distance_functions
        self.distance_function: str = distance_function.lower()
        self.supported_backbones = {"4conv", "resnet18"}
        assert backbone in self.supported_backbones

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        if backbone == "resnet18":
            assert self.pooling is None, "resnet impl handles pooling"
            self.encoder = build_resnet18_encoder()
        elif backbone == "4conv":
            self.encoder = build_4conv_protonet_encoder(
                in_channels, hidden_size, out_channels
            )
        else:
            raise ValueError(
                f"Unsupported backbone {backbone} not in {self.supported_backbones}"
            )

        self.prototypes: Dict[
            int, torch.Tensor
        ] = {}  # Map from class index to prototype
        self.examples_per_class: Dict[
            int, int
        ] = {}  # Map from class index to n examples seen thus far

    def forward(self, inputs):
        inputs_reshaped = inputs.view(
            -1, *inputs.shape[2:]
        )  # -> [b * k * n, ch, rows, cols]
        embeddings = self.encoder(inputs_reshaped)  # -> [n * k, ch, rows, cols]
        embeddings_reshaped = embeddings.view(
            *inputs.shape[:2], -1
        )  # -> [n * k, ch * rows * cols]
        return embeddings_reshaped

    def evaluate(
        self,
        train_inputs,
        train_labels,
        test_inputs,
        test_labels,
        n_train_classes: int,
        k_shots: int,
    ) -> torch.FloatTensor:
        """Computes the mean accuracy of predictions on test_inputs."""

        # print("training data")
        # plot_image_batch(train_inputs[0], train_labels[0])
        # print("test data")
        # plot_image_batch(test_inputs[0], test_labels[0])
        train_embeddings = self.forward(train_inputs)
        test_embeddings = self.forward(test_inputs)
        all_class_prototypes = self._get_prototypes(
            train_embeddings, train_labels, n_train_classes, k_shots
        )
        if self.distance_function == "euclidean":
            mahala = False
        elif self.distance_function == "mahalanobis":
            mahala = True
        else:
            raise NotImplementedError(
                f"Distance function must be in {self.supported_distance_functions} but is {self.distance_function}"
            )

        acc = get_protonet_accuracy(
            all_class_prototypes, test_embeddings, test_labels, mahala=mahala
        )
        return acc

    def _get_prototypes(
        self,
        train_embeddings: torch.Tensor,
        train_labels: torch.Tensor,
        n_train_classes: int,
        k_shots: int,
    ):
        new_prototypes = get_prototypes(
            train_embeddings, n_train_classes, k_shots
        )  # -> [b, n, features]

        # Add each prototype to the model:
        assert len(new_prototypes.shape) == 3
        assert new_prototypes.shape[0] == 1
        new_prototypes = new_prototypes[
            0
        ]  # Assume single element in batch dimension. Now [n_train_classes, features]
        # class_indices = np.unique(train_labels.numpy())
        class_indices = train_labels.unique()
        # class_indices = torch.unique(train_labels)
        for i, cls_index in enumerate(class_indices):
            self.update_prototype_for_class(
                cls_index, new_prototypes[i, :], train_labels.shape[1]
            )
        all_class_prototypes = [
            self.prototypes[key] for key in sorted(self.prototypes.keys())
        ]
        all_class_prototypes: torch.FloatTensor = torch.stack(
            all_class_prototypes, dim=0
        ).unsqueeze(
            0
        )  # -> [1, n, features]
        return all_class_prototypes

    def update_prototype_for_class(
        self, class_index: int, proto: torch.FloatTensor, n_samples: int
    ):
        """
        Computes unbiased, online average prototypes for each class_index.
        if self.distance_function = "mahalanobis", it is assumed that the second half of the prototype vector is the predicted standard deviation
        """
        assert len(proto.shape) == 1
        with torch.no_grad():
            try:
                old_proto = self.prototypes[class_index]
                self.examples_per_class[class_index] += n_samples
                self.prototypes[class_index] = old_proto + (
                    (proto - old_proto) / self.examples_per_class[class_index]
                )
            except KeyError:
                self.prototypes[class_index] = proto
                self.examples_per_class[class_index] = n_samples


def get_num_samples(targets, num_classes, dtype=None):
    """Returns tensor of shape batch by num_classes of num samples per class"""
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)  # -> [b, n]
    return num_samples


def get_prototypes_unsorted(embeddings, targets, num_classes):
    """Compute the prototypes (the mean vector of the embedded training/support
    points belonging to its class) for each class in the task.
    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor
        has shape `(batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has
        shape `(batch_size, num_examples)`.
    num_classes : int
        Number of classes in the task.
    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    """
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)

    num_samples = get_num_samples(targets, num_classes, dtype=embeddings.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
    indices = targets.unsqueeze(-1).expand_as(embeddings)
    prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)

    return prototypes


def get_prototypes(
    embeddings,
    n_classes,
    k_shots_per_class,
    return_sd: bool = False,
    prototype_normal_std_noise: Optional[float] = None,
):
    """Compute the prototypes (the mean vector of the embedded training/support
    points belonging to its class) for each class in the task.

    NOTE:
        THIS IMPLEMENTATION ASSUMES THAT EMBEDDINGS AND TARGETS ARE SORTED SUCH THAT CLASS EXAMPLES ARE CONTIGUOUS AND IN CORRESPONDING ORDER ALONG THE SECOND AXIS OF `embeddings`.

    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor
        has shape `(batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has
        shape `(batch_size, num_examples)`.
    num_classes : int
        Number of classes in the task.
    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    """
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)
    embeddings_reshaped = embeddings.reshape(
        [batch_size, n_classes, k_shots_per_class, embedding_size]
    )
    prototypes = embeddings_reshaped.mean(2)
    # print(f"Prototype shape for {n_classes} [batch, n_classes, embedding_size]: {prototypes.shape}")
    assert len(prototypes.shape) == 3

    if prototype_normal_std_noise is not None:
        prototypes += torch.normal(
            torch.zeros_like(prototypes),
            torch.ones_like(prototypes) * prototype_normal_std_noise,
        )
    if return_sd:
        # Compute standard deviation across embeddings. This doesn't really extend to 1 shot case, so we default to False
        #   and assume that usually standard deviation or covariance is predicted by the neural network directly.
        return prototypes, embeddings_reshaped.std(2)
    return prototypes


def prototypical_loss(
    prototypes, embeddings, targets, sum_loss_over_examples, **kwargs
):
    """Compute the loss (i.e. negative log-likelihood) for the prototypical
    network, on the test/query points.
    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(batch_size, num_examples)`.
    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    """
    squared_distances = torch.sum(
        (prototypes.unsqueeze(2) - embeddings.unsqueeze(1)) ** 2, dim=-1
    )
    # squared_distances.shape
    # torch.Size([1, 5, 25])
    # targets.shape
    # torch.Size([1, 25])
    if sum_loss_over_examples:
        reduction = "sum"
    else:
        reduction = "mean"
    return F.cross_entropy(-squared_distances, targets, reduction=reduction, **kwargs)


def gaussian_prototypical_mahalanobis_loss(prototypes, embeddings, targets, **kwargs):
    """Compute the loss (i.e. negative log-likelihood) for the prototypical
    network, on the test/query points.
    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(batch_size, num_examples)`.
    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    """
    embedding_dim = prototypes.shape[-1]
    prototypes, sd = (
        prototypes[:, :, 0 : embedding_dim // 2],
        prototypes[:, :, embedding_dim // 2 :],
    )
    sd = sd.mul(0.5).exp_()
    embeddings = embeddings[:, :, 0 : embedding_dim // 2]
    sq_distances = torch.sum(
        ((prototypes.unsqueeze(1) - embeddings.unsqueeze(2)) ** 2) / sd, dim=-1
    )
    sq_distances = sq_distances.permute(0, 2, 1)
    return F.cross_entropy(-sq_distances, targets, **kwargs).unsqueeze(0)


def get_protonet_accuracy(
    prototypes: torch.FloatTensor,
    embeddings: torch.FloatTensor,
    targets: Union[torch.Tensor, torch.LongTensor],
    jsd: bool = False,
    mahala: bool = False,
) -> torch.FloatTensor:
    """Compute the accuracy of the prototypical network on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(meta_batch_size, num_examples)`.
    jsd: bool. If True, only use first half of prototypes and embeddings.


    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    if mahala or jsd:
        embedding_dim = prototypes.shape[-1]
        prototypes, sd = (
            prototypes[:, :, 0 : embedding_dim // 2],
            prototypes[:, :, embedding_dim // 2 :],
        )
        sd = sd.mul(0.5).exp_()
        embeddings = embeddings[:, :, 0 : embedding_dim // 2]
        sq_distances = torch.sum(
            ((prototypes.unsqueeze(1) - embeddings.unsqueeze(2)) ** 2) / sd, dim=-1
        )
    else:
        sq_distances = torch.sum(
            (prototypes.unsqueeze(1) - embeddings.unsqueeze(2)) ** 2, dim=-1
        )
    _, predictions = torch.min(sq_distances, dim=-1)
    return get_accuracy(predictions, targets)
