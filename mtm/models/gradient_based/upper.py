from typing import List, Dict

from torch.utils.data.dataset import Dataset
from torch import nn
import torch
from torchvision import models
from torch.utils.data.dataloader import DataLoader
from torch import optim
from tqdm import tqdm
from mtm.models.gradient_based.protonet_cnn import ProtonetCNN
from torch.optim.lr_scheduler import StepLR


class Upper(nn.Module):
    def __init__(self, n_classes, model_config):
        """
        Upper model class
        """
        super(Upper, self).__init__()
        self.n_classes = n_classes
        self.config = model_config

        if self.config["model"] == "resnet18":
            self.model = models.resnet18(pretrained=False)
            input_features = self.model.fc.in_features
            self.model.fc = nn.Linear(input_features, n_classes, bias=False)
        elif self.config["model"] == "ProtonetCNN":
            self.model = ProtonetCNN(n_classes=n_classes)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        # dict for label to neuron in fc mapping
        self.mapping = {}
        self.add_base_classes_to_mapping()

        # store all the examples
        # this will be indexed by int_class_id
        # used in server to keep all the training images
        self.exemplar_sets = {}

        self.best_update_epochs = None

        self.merge_count = 0

        # this will be indexed by ext_class_id
        # use this during model update
        # and server uses them during merge
        self.mission_exemplar_sets = {}

        print("[Model] Upper initialized...")

    def add_base_classes_to_mapping(self):
        for i in range(self.n_classes):
            self.mapping[i] = i

    def extract_features(self, x):
        model = nn.Sequential(*list(self.model.children())[:-1])
        x = model(x)
        return x

    def forward(self, x):
        """
        pass input tensor to model
        :param x: input tensor
        :return: output from network
        """
        x = self.model(x)
        return x

    def add_class(self, new_class):
        """
        Add new class
        :param new_class: new class to add
        :return: None
        """
        if new_class not in self.mapping.keys():
            self.mapping[new_class] = self.n_classes
        else:
            # weight values already initialized return
            return

        in_features = self.model.fc.in_features
        previous_classes = self.model.fc.out_features
        previous_weights = self.model.fc.weight.data
        self.model.fc = nn.Linear(in_features, previous_classes + 1, bias=False)
        self.model.fc.weight.data[:previous_classes] = previous_weights
        self.n_classes += 1

    def train_base_model(self, base_train_dataset: Dataset):
        """
        :param base_train_dataset: train the model with base_train_dataset
        :return: None
        """
        if self.config["model"] == "resnet18":
            self.best_update_epochs = self.config["best_epoch_resnet18"]
        elif self.config["model"] == "ProtonetCNN":
            self.best_update_epochs = self.config["best_epoch_protonetCNN"]

        if self.best_update_epochs is None:
            print("Please provide best_epoch to update parameters")
            raise KeyError("[Model] best_update_epochs not set in config file")

        if self.config["model"] == "resnet18":
            base_model_path = self.config["base_model_path_resnet18"]
        else:
            base_model_path = self.config["base_model_path_protonet"]

        try:
            self.model.load_state_dict(torch.load(base_model_path))
            print(f"[Model] loaded model from {base_model_path}")
        except FileNotFoundError:
            print(f"Unable to find trained model at {base_model_path}")
            raise FileNotFoundError("[Model] Please provide path to trained model")

    def update_base_memory(self, base_train_dataset):
        batch_size = self.config["batch_size_base_memory"]
        num_workers = self.config["num_workers"]

        train_loader = DataLoader(
            base_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )

        # create exemplars and add to them
        for class_id in range(self.n_classes):
            self.exemplar_sets[class_id] = []

        print("[Model] Creating exemplar set for base classes")
        # This will be called from the client_server once so let's use int_class_id
        for inputs, labels in train_loader:
            for idx, label in enumerate(labels):
                self.exemplar_sets[label.item()].append(inputs[idx])

    def test_model(self, test_dataset: Dataset):
        """
        :param test_dataset: test_dataset to test against
        :return: test accuracy
        """
        batch_size = self.config["base_test_batch_size"]
        num_workers = self.config["num_workers"]

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )

        correct = 0.0
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[Model] testing on test dataset")
        with torch.no_grad():
            for data, label in test_loader:
                # change label for field classes
                for idx, lab in enumerate(label):
                    if lab.item() in self.mapping.keys():
                        label[idx] = self.mapping[lab.item()]

                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                pred = torch.argmax(output, dim=1)
                correct += (pred == label).sum().item()

        acc = correct / len(test_dataset)
        return acc

    def update_model(
        self,
        mission_data: torch.Tensor,
        mission_labels: torch.Tensor,
        mission_classes: List[int],
    ):
        """
        :param mission_data: mission data [will be dataset later, tensor for now]
        :param mission_labels: mission_labels
        :param mission_classes: new classes learned from mission
        :return: None
        """

        lr = self.config["lr_update_model"]
        batch_size = self.config["batch_size_update_model"]

        print(f"[Model] updating model")
        for new_class in mission_classes:
            if new_class not in self.mapping.keys():
                self.add_class(new_class)

        self.model = self.model.to(self.device)
        self.model.train()

        # update the mission_exemplar_sets
        print("[Model] updating exemplar sets")
        # print(f"mission_classes: {mission_classes}")
        # print(f"mission_labels.unique().tolist(): {mission_labels.unique().tolist()}")

        assert set(mission_classes) == set(mission_labels.unique().tolist())
        for mission_class in mission_classes:
            self.mission_exemplar_sets[mission_class] = []

        for idx, item in enumerate(mission_data):
            self.mission_exemplar_sets[mission_labels[idx].item()].append(item)

        if self.config["update_client"]:
            if self.config["optimizer_update_model"] == "Adam":
                optimizer = optim.Adam(self.model.parameters(), lr=lr)
            else:
                optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=lr,
                    momentum=0.9,
                    weight_decay=1e-3,
                    nesterov=True,
                )

            criterion = nn.CrossEntropyLoss()
            # create new tensor dataset from exemplar dataset and mission_data
            x_train, y_train = [], []
            for class_id, tensor_list in self.exemplar_sets.items():
                for item in tensor_list:
                    x_train.append(item)
                    y_train.append(class_id)

            for idx, item in enumerate(mission_data):
                x_train.append(item)
                # change label according to current mapping
                y_train.append(self.mapping[mission_labels[idx].item()])

            x_train = torch.stack(x_train)
            y_train = torch.tensor(y_train)
            train_dataset = torch.utils.data.dataset.TensorDataset(x_train, y_train)
            train_loader = torch.utils.data.dataloader.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

            for epoch in tqdm(range(self.best_update_epochs)):
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                scheduler.step(epoch=epoch)
        else:
            return

    def eval_model(self, eval_data: torch.Tensor, eval_labels: torch.Tensor):
        """
        :param eval_data: tensor for now, will be a dataset later
        :param eval_labels: tensor for now, will be a dataset later
        :return:
        """

        correct = 0.0
        self.model.eval()
        print(f"[Model] evaluating on new classes")
        batch_size = eval_data.shape[0]
        labels = torch.tensor([0] * batch_size)
        for idx, item in enumerate(eval_labels):
            labels[idx] = self.mapping[item.item()]
        data, labels = eval_data.to(self.device), labels.to(self.device)
        tensor_dataset = torch.utils.data.TensorDataset(data, labels)
        test_loader = torch.utils.data.dataloader.DataLoader(
            tensor_dataset, batch_size=64, shuffle=False
        )
        with torch.no_grad():
            for inputs, labels in test_loader:
                output = self.model(inputs)
                pred = torch.argmax(output, dim=1)
                correct += (pred == labels).sum().item()

        acc = correct / eval_labels.shape[0]
        return acc

    def merge_models(
        self,
        new_models: Dict[int, List[torch.Tensor]],
        new_classes: Dict[int, List[int]],
    ):
        """
        :param new_models: new models learned by clients from mission, dictionary of client id and thier learned models in other
        cases like IcaRL, Protonets, Lower
        for Upper method:
        there won't be learned model but the Dict of label as the key and list of Tensors
        :param new_classes: new classes learned by clients from mission, Dict of client_id and list of new_classes learned
        :return: None
        """
        print("[Model] merging models")

        batch_size = self.config["batch_size_merge"]
        epochs = self.config["epochs_merge"]
        lr = self.config["lr_merge"]

        # update mapping
        num_new: int = 0
        for client_id, mission_classes in new_classes.items():
            for mission_class in mission_classes:
                if mission_class not in self.mapping.keys():
                    self.mapping[mission_class] = self.n_classes
                    self.n_classes += 1
                    num_new += 1

        if self.config["start_from_scratch"]:
            # start from scratch
            if self.config["model"] == "resnet18":
                in_features = self.model.fc.in_features
                self.model = models.resnet18(pretrained=False)
                self.model.fc = nn.Linear(in_features, self.n_classes, bias=False)
            else:
                self.model = ProtonetCNN(n_classes=self.n_classes)
        else:
            # start from pretrained models
            # add new units in fc for new classes
            if num_new > 0:
                if self.config["model"] == "resnet18":
                    in_features = self.model.fc.in_features
                    previous_classes = self.model.fc.out_features
                    previous_weights = self.model.fc.weight.data
                    self.model.fc = nn.Linear(
                        in_features, previous_classes + num_new, bias=False
                    )
                    self.model.fc.weight.data[:previous_classes] = previous_weights
                else:
                    in_features = self.model.fc.in_features
                    previous_classes = self.model.fc.out_features
                    previous_weights = self.model.fc.weight.data
                    self.model.fc = nn.Linear(
                        in_features, previous_classes + num_new, bias=False
                    )
                    self.model.fc.weight.data[:previous_classes] = previous_weights

        # update the exemplar sets
        for client_id, class_list in new_classes.items():
            for class_id in class_list:
                server_label = self.mapping[class_id]
                if server_label not in self.exemplar_sets.keys():
                    self.exemplar_sets[server_label] = []
                    self.exemplar_sets[server_label].extend(
                        new_models[client_id].mission_exemplar_sets[class_id]
                    )
                else:
                    if len(self.exemplar_sets[server_label]) > 500:
                        raise Exception("[Model] ")
                    else:
                        self.exemplar_sets[server_label].extend(
                            new_models[client_id].mission_exemplar_sets[class_id]
                        )

        # after model merging - set the mission_exemplar_sets to empty again so that
        # when client collects examples, it will start fresh and previous duplicatio will not happen
        self.mission_exemplar_sets = {}
        # train on whole new dataset combined on all the examples seen so far
        self.model = self.model.to(self.device)
        self.model.train()

        if self.config["optimizer_merge"] == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=1e-3,
                nesterov=True,
            )

        criterion = nn.CrossEntropyLoss()
        # create new tensor dataset from exemplar dataset and mission_data
        x_train, y_train = [], []
        for class_id, tensor_list in self.exemplar_sets.items():
            for item in tensor_list:
                x_train.append(item)
                y_train.append(class_id)

        x_train = torch.stack(x_train)
        y_train = torch.tensor(y_train)
        train_dataset = torch.utils.data.dataset.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.dataloader.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
        for epoch in tqdm(range(epochs)):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step(epoch=epoch)

        print("[Model] models merged")
