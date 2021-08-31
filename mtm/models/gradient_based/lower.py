from torch.utils.data.dataset import Dataset
from torch import nn
import torch
from torchvision import models
from torch.utils.data.dataloader import DataLoader
from torch import optim
from tqdm import tqdm
from mtm.models.gradient_based.protonet_cnn import ProtonetCNN
from torch.optim.lr_scheduler import StepLR


class Lower(nn.Module):
    def __init__(self, n_classes, model_config):
        """
        Lower model class
        :param n_classes: number of classes
        """
        super(Lower, self).__init__()
        self.n_classes = n_classes
        self.config = model_config

        if self.config["model"] == "resnet18":
            self.model = models.resnet18(pretrained=False)
            input_features = self.model.fc.in_features
            self.model.fc = nn.Linear(input_features, n_classes, bias=False)
        elif self.config["model"] == "ProtonetCNN":
            self.model = ProtonetCNN(n_classes=n_classes)

        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        # dict for label to neuron in fc mapping
        self.mapping = {}
        self.add_base_classes_to_mapping()

        self.best_update_epochs = None
        print("[Model] Lower initialized...")

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
        pass

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

    def update_model(self, mission_data, mission_labels, mission_classes):
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

        self.model.train()
        self.model = self.model.to(self.device)

        if self.config["optimizer_update_model"] == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=1e-3,
            )

        criterion = nn.CrossEntropyLoss()
        # create new tensor dataset from exemplar dataset and mission_data
        x_train, y_train = [], []

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

    def merge_models(self, new_models, new_classes):
        """
        :param new_models: new models learned by clients from mission
        :param new_classes: new classes learned by clients from mission
        :return: None
        """
        print("[Model] merging models")
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
                    class_id, new_models[client_id].model.fc.weight.data[client_label]
                )

        print("[Model] models merged")
