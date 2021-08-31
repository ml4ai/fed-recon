from torch.utils.data.dataset import Dataset
from torch import nn
import torch
import torch.nn.functional as f
from torchvision import models
from torch.utils.data.dataloader import DataLoader
from torch import optim
from tqdm import tqdm
from copy import deepcopy
from fed_recon.util.util import to_one_hot
from fed_recon.models.gradient_based.protonet_cnn import ProtonetCNN
import os
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt


class ICaRL(nn.Module):
    def __init__(self, n_classes, model_config):
        """
        IcaRL model class
        :param n_classes: number of classes
        """
        super(ICaRL, self).__init__()
        self.n_classes = n_classes
        self.config = model_config
        self.budget = self.config["budget"]

        if self.config["model"] == "resnet18":
            self.model = models.resnet18(pretrained=False)
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)
            input_features = self.model.fc.in_features
            self.model.fc = nn.Linear(input_features, n_classes, bias=False)
        elif self.config["model"] == "ProtonetCNN":
            self.model = ProtonetCNN(n_classes=n_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        # dict for label to neuron in fc mapping
        self.mapping = {}
        self.add_base_classes_to_mapping()

        # store all the examples
        # this will be indexed by int_class_id
        # used in server to keep all the training images
        self.exemplar_sets = {}

        self.means = {}

        self.best_update_epochs = None

        # this will be indexed by ext_class_id
        # use this during model update
        # and server uses them during merge
        self.mission_exemplar_sets = {}

        print("[Model] IcaRL initialized...")

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

    def reduce_exemplar_set(self, m):
        """
        keep upto m examples
        :param m: keep m examples per class
        :return: None
        """
        for class_id, examples in self.exemplar_sets.items():
            if len(examples) > m:
                self.exemplar_sets[class_id] = examples[:m]

    def compute_means(self):
        """
        compute the mean of exemplars from current model
        :return: none
        """
        for class_id, exemplars in self.exemplar_sets.items():
            examples = torch.stack(exemplars).to(self.device)
            with torch.no_grad():
                features = self.extract_features(examples)
            features = f.normalize(features, p=2, dim=1)
            mean = features.mean(dim=0, keepdim=True)
            mean = f.normalize(mean, p=2, dim=1)
            self.means[class_id] = mean.squeeze()

    def classify(self, x):
        """
        classification based on the means of exemplars
        :param x: input batch
        :return: predictions
        """
        preds = []
        features = self.extract_features(x)
        features = f.normalize(features, p=2, dim=1)
        dist = {}
        for feature in features:
            for class_id, mean in self.means.items():
                dist[class_id] = [torch.sum(feature - self.means[class_id]) ** 2]
            pred = min(dist, key=dist.get)
            preds.append(pred)

        preds = torch.tensor(preds)
        return preds

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

    def add_class_weight(self, new_class, weight):
        """
        Add new class and put the weight
        :param new_class: new class to add
        :param weight: weight for that class
        :return: None
        """
        if new_class not in self.mapping.keys():
            self.mapping[new_class] = self.n_classes
            in_features = self.model.fc.in_features
            previous_classes = self.model.fc.out_features
            previous_weights = self.model.fc.weight.data
            self.model.fc = nn.Linear(in_features, previous_classes + 1, bias=False)
            self.model.fc.weight.data[:previous_classes] = previous_weights
            self.model.fc.weight.data[-1] = weight
            self.n_classes += 1
        else:
            idx = self.mapping[new_class]
            self.model.fc.weight.data[idx] = (
                self.model.fc.weight.data[idx] + weight.cpu()
            ) / 2.0

    def train_model(self, base_train_dataset, base_val_dataset, save_model=False):
        num_epochs = self.config["epochs_base_train"]
        batch_size = self.config["batch_size_base_train"]
        lr = self.config["lr_base_train"]
        num_workers = self.config["num_workers"]

        if self.config["model"] == "resnet18":
            base_model_path = self.config["base_model_path_resnet18"]
        else:
            base_model_path = self.config["base_model_path_protonet"]

        train_loader = DataLoader(
            base_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )

        val_loader = DataLoader(
            base_val_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )

        best_model = None
        best_acc = 0.0

        if self.config["optimizer_base_train"] == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=1e-2,
                nesterov=True,
            )

        criterion = nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

        print("[Model] Training base model...")
        self.model.train()
        best_epoch = 0

        train_acc_list = []
        val_acc_list = []
        val_loss_list = []
        train_loss_list = []

        for epoch in tqdm(range(num_epochs)):

            train_correct = 0
            train_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                preds = torch.argmax(outputs, dim=1).detach()
                train_correct += (preds == labels).sum().item()
                train_loss += loss.item()

            val_correct = 0
            val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            train_acc = train_correct / len(base_train_dataset)
            val_acc = val_correct / len(base_val_dataset)
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_model = deepcopy(self.model)

            scheduler.step(epoch=epoch)

        # save the model
        self.model.load_state_dict(best_model.state_dict())
        print(f"[Model] best_epoch: {best_epoch}")
        print(f"[Model] best_acc: {best_acc}")
        if save_model:
            if os.path.exists(base_model_path):
                os.remove(base_model_path)

            torch.save(self.model.state_dict(), base_model_path)
            print(f"[Model] best model saved to: {base_model_path}")

        # plot the graph
        plt.plot(list(range(num_epochs)), train_acc_list, label="train_acc")
        plt.plot(list(range(num_epochs)), val_acc_list, label="val_acc")
        plt.legend()
        plt.show()

        plt.plot(list(range(num_epochs)), train_loss_list, label="train_loss")
        plt.plot(list(range(num_epochs)), val_loss_list, label="val_loss")
        plt.legend()
        plt.show()

    def tune_hyperparameters(
        self, base_train_dataset, base_val_dataset, hold_train_dataset, hold_val_dataset
    ):
        epochs = self.config["epochs_update_model"]
        lr = self.config["lr_update_model"]
        batch_size = self.config["batch_size_update_model"]
        num_workers = self.config["num_workers"]

        # first train on base_train and base_val and select best model
        self.train_model(base_train_dataset, base_val_dataset, save_model=False)
        # train on hold out dataset and evaluate on hold out val dataset
        # add 10 classes
        new_classes = list(range(40, 50))
        for new_class in new_classes:
            self.add_class(new_class)

        self.model = self.model.to(self.device)

        hold_out_train_loader = DataLoader(
            hold_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )

        hold_out_val_loader = DataLoader(
            hold_val_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )

        print("[Model] Training model for hyper-parameter tuning")

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

        best_epoch = 0
        best_acc = 0.0

        criterion = nn.CrossEntropyLoss()
        self.model.train()
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

        for epoch in tqdm(range(epochs)):
            for inputs, labels in hold_out_train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            correct = 0
            with torch.no_grad():
                for inputs, labels in hold_out_val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()

            acc = (correct / len(hold_val_dataset)) * 100.0
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch

            scheduler.step(epoch=epoch)

        print(f"[Model] best epoch: {best_epoch}")
        print(f"[Model] best_acc: {best_acc}")
        tune_location = self.config["tune_location"]
        model_name = self.config["model"]

        file_path = tune_location + "/" + model_name + ".txt"
        print(f"[Model] writing best epoch to {file_path}")

        with open(file_path, "w") as write_file:
            write_file.write(f"best epoch: {best_epoch}\n")
        print("[Model] hyper parameter optimization done")
        exit()

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
            self.model.load_state_dict(
                torch.load(base_model_path, map_location="cuda:0")
            )
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
        examples_per_class = int(self.budget / self.n_classes)
        for inputs, labels in train_loader:
            for idx, label in enumerate(labels):
                if len(self.exemplar_sets[label.item()]) < examples_per_class:
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
                    else:
                        raise Exception(
                            f"[Model] no mapping for class_id: {lab.item()}"
                        )

                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                pred = torch.argmax(output, dim=1)
                correct += (pred == label).sum().item()

        acc = correct / len(test_dataset)
        return acc

    def update_model(self, mission_data, mission_labels, mission_classes):
        """
        :param mission_data: mission data [will be dataset later, tensor for now]
        :param mission_labels: mission_labels
        :param mission_classes: new classes learned from mission
        :return: None
        """
        previous_model = deepcopy(self.model).eval()
        lr = self.config["lr_update_model"]
        batch_size = self.config["batch_size_update_model"]

        print(f"[Model] updating model")
        num_new_classes = 0
        for new_class in mission_classes:
            if new_class not in self.mapping.keys():
                self.add_class(new_class)
                num_new_classes += 1

        self.model = self.model.to(self.device)
        self.model.train()

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

        # criterion = nn.CrossEntropyLoss()
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

        for _ in tqdm(range(self.best_update_epochs)):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # calculate scores for distillation loss
                with torch.no_grad():
                    scores = previous_model(inputs)

                outputs = self.model(inputs)
                binary_targets = to_one_hot(labels.cpu(), outputs.size(1)).to(
                    labels.device
                )
                if num_new_classes > 0:
                    binary_targets = binary_targets[:, -num_new_classes:]
                    binary_targets = torch.cat(
                        [
                            torch.sigmoid(scores / self.config["temperature"]),
                            binary_targets,
                        ],
                        dim=1,
                    )
                if outputs.shape != binary_targets.shape:
                    raise Exception(
                        "[Model]: output and binary target shapes do not match"
                    )

                loss = (
                    f.binary_cross_entropy_with_logits(
                        input=outputs, target=binary_targets, reduction="none"
                    )
                    .sum(dim=1)
                    .mean()
                )
                # loss = criterion(outputs, labels)
                optimizer.zero_grad()
                self.model.zero_grad()
                loss.backward()
                optimizer.step()

        # update the exemplar sets
        assert set(mission_classes) == set(mission_labels.unique().tolist())
        for mission_class in mission_classes:
            self.mission_exemplar_sets[mission_class] = []

        for idx, item in enumerate(mission_data):
            self.mission_exemplar_sets[mission_labels[idx].item()].append(item)

        del previous_model

    def eval_model(self, eval_data, eval_labels):
        """
        :param eval_data: tensor for now, will be a dataset later
        :param eval_labels: tensor for now, will be a dataset later
        :return:
        """
        correct = 0.0
        print(f"[Model] evaluating on new classes")
        batch_size = eval_data.shape[0]
        labels = torch.tensor([0] * batch_size)
        for idx, item in enumerate(eval_labels):
            labels[idx] = self.mapping[item.item()]

        data, labels = eval_data.to(self.device), labels.to(self.device)
        if self.config["test_method"] == "exemplars":
            print("[Model] Eval with exemplar means")
            self.compute_means()
            with torch.no_grad():
                pred = self.classify(data).to(self.device)
                correct += (pred == labels).sum().item()
        else:
            print("[Model] Eval with softmax")
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

    def merge_models(self, new_models, new_classes):
        """
        :param new_models: new models learned by clients from mission
        :param new_classes: new classes learned by clients from mission
        :return: None
        """
        print("[Model] merging models")

        batch_size = self.config["batch_size_merge"]
        epochs = self.config["epochs_merge"]
        lr = self.config["lr_merge"]

        if self.config["old_style_merge"]:
            # find common classes
            common_classes = self.model.fc.out_features
            # average out the common model parts: this was to average the previous model as well
            # m = self.model.state_dict()
            # average from new models only without old model
            first = True
            for client_id, item in new_models.items():
                if first:
                    m = item.model.state_dict()
                    for name, value in m.items():
                        if name == "fc.weight":
                            m[name] = m[name][:common_classes, :]
                    first = False
                    continue

                temp = item.model.state_dict()
                for name, value in temp.items():
                    if name == "fc.weight":
                        # m[name] = (m[name] + temp[name][:common_classes, :]) / 2.0
                        m[name] = m[name] + temp[name][:common_classes, :]
                    else:
                        # m[name] = (m[name] + temp[name]) / 2.0
                        m[name] = m[name] + temp[name]

            total_models_merged = len(new_models)
            for name, value in m.items():
                if name == "fc.weight":
                    m[name] = m[name] / total_models_merged
                else:
                    m[name] = m[name] / total_models_merged

            if self.config["model"] == "resnet18":
                self.model = models.resnet18(pretrained=False)
                self.model.avgpool = nn.AdaptiveAvgPool2d(1)
                input_features = self.model.fc.in_features
                self.model.fc = nn.Linear(input_features, common_classes, bias=False)
                self.model.load_state_dict(m)
            else:
                self.model = ProtonetCNN(n_classes=common_classes)
                self.model.load_state_dict(m)

            # now append the remaining parts for fc layer
            for client_id, class_list in new_classes.items():
                for class_id in class_list:
                    client_label = new_models[client_id].mapping[class_id]
                    self.add_class_weight(
                        class_id,
                        new_models[client_id].model.fc.weight.data[client_label],
                    )
        else:
            # update mapping
            num_new: int = 0
            for client_id, mission_classes in new_classes.items():
                for mission_class in mission_classes:
                    if mission_class not in self.mapping.keys():
                        self.mapping[mission_class] = self.n_classes
                        self.n_classes += 1
                        num_new += 1

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
                        raise Exception("[Model] exemplars replication")
                    else:
                        self.exemplar_sets[server_label].extend(
                            new_models[client_id].mission_exemplar_sets[class_id]
                        )

        # after model merging - set the mission_exemplar_sets to empty again so that
        # when client collects examples, it will start fresh and previous duplicatio will not happen
        self.mission_exemplar_sets = {}

        # reduce the exemplars in case more
        examples_per_class = int(self.budget / self.n_classes)
        self.reduce_exemplar_set(examples_per_class)

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

        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
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
