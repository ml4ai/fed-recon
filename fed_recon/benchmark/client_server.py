import random
import warnings
from copy import deepcopy
from typing import Optional, Tuple, Union, Set, List

import numpy as np
import torch

from fed_recon.benchmark.mini_imagenet import MiniImagenet
from fed_recon.util.experiment import ExperimentLogger


class ModelUpdatingException(Exception):
    pass


class Client:
    """ Initialize the Client """

    def __init__(self, client_id: int, client_name: str, environment: MiniImagenet):
        self.id = client_id
        self.name = client_name
        self.environment = environment
        self.model = None
        # classes learned till now
        self.classes = []

        self.mission_data: Optional[torch.Tensor] = None
        self.mission_labels: Optional[torch.Tensor] = None
        self.mission_classes: Optional[List[int]] = None

    def clean(self):
        """ Clean the memory """
        self.model = None
        # classes learned till now
        self.classes = []

        self.mission_data = None
        self.mission_labels = None
        self.mission_classes = None

    def go_to_mission(self, model):
        # FIXME: rename this method to get_mission_data
        #   all the func is doing is getting data
        #   go_to_mission is also confusing since we assume on client learning but no learning is happening here
        """ Get mission data to learn new classes """
        print(f"[Client {self.id}] going to mission")
        self.model = model
        # before going to mission initialize the new_classes to empty again
        print(f"[Client {self.id}] requesting mission data from Environment")
        (
            self.mission_data,
            self.mission_labels,
        ) = self.environment.get_mission_data_train_tensor()
        if self.mission_labels is not None:
            self.mission_classes = list(set(self.mission_labels.unique().tolist()))
        else:
            self.mission_classes = None
        print(f"[Client {self.id}] got mission data from Environment")

    def update_model(self):
        """ Update model during a return from a mission """
        print(f"[Client {self.id}] updating model")
        self.model.update_model(
            self.mission_data, self.mission_labels, self.mission_classes
        )

    def give_model_to_server(self):
        """ Give updated model to server """
        print(f"[Client {self.id}] providing new model to Server")
        return deepcopy(self.model)

    def give_new_classes_to_server(self):
        """ Give the classes it learned to Server """
        print(f"[Client {self.id}] providing new classes to Server")
        return self.mission_classes

    def eval_model_base_classes(self) -> float:
        """ Eval new model on base classes """
        print(f"[Client {self.id}] evaluating model on base classes")
        eval_data, eval_labels = self.environment.get_base_data_test_tensor(
            self.environment.base_labels
        )
        acc = self.model.eval_model(eval_data, eval_labels)
        if isinstance(acc, torch.Tensor):
            acc = float(acc.item())
        print(f"[Client] Test accuracy on base classes: {round(acc, 2)}")
        return acc

    def eval_model_reconned_classes(self, reconned_classes: Set[str]) -> float:
        """Eval the model on reconned classes"""
        print(f"[Client {self.id}] evaluating model on reconned classes")
        eval_data, eval_labels = self.environment.get_mission_data_test_tensor(
            reconned_classes
        )
        acc = self.model.eval_model(eval_data, eval_labels)
        if isinstance(acc, torch.Tensor):
            acc = float(acc.item())
        print(
            f"[Client] Test accuracy on reconned classes {self.mission_classes}: {round(acc, 2)}"
        )
        return acc

    def eval_model_new_classes(self) -> float:
        """ Eval new model on newly learned classes """
        print(f"[Client {self.id}] evaluating model on new classes")
        eval_data, eval_labels = self.environment.get_mission_data_test_tensor(
            self.mission_classes
        )
        acc = self.model.eval_model(eval_data, eval_labels)
        if isinstance(acc, torch.Tensor):
            acc = float(acc.item())
        print(
            f"[Client] Test accuracy on new classes from mission {self.mission_classes}: {round(acc, 2)}"
        )
        return acc

    def evalute_client_model_on_mission(
        self, reconned_classes: Set[str]
    ) -> Tuple[float, float, float]:
        """
        Clients will update_model then evaluate on the data from the mission

        This is the Eval 1 diamond in fed recon diagram
        """
        print(f"[Client {self.id}] updating model on mission")
        # update new classes learned
        self.classes.extend(self.mission_classes)
        self.update_model()
        print(f"[Client {self.id}] evaluating model on mission")
        print(f"self.mission_classes {self.mission_classes}")
        base_acc = self.eval_model_base_classes()
        if len(reconned_classes) > 0:
            recon_acc = self.eval_model_reconned_classes(reconned_classes)
        else:
            recon_acc = None
        new_classes_acc = self.eval_model_new_classes()
        return base_acc, recon_acc, new_classes_acc


def record_eval_0(acc: float, experiment: ExperimentLogger):
    fn = "eval_0_base_test.csv"
    cols = ("task_set", "dataset", "accuracy")
    row = ["base", "test", acc]
    experiment.record_result(fn, cols, row)
    return experiment


def record_eval_1(
    mission_number: int,
    client_id: Union[str, int],
    n_clients: int,
    n_base_classes,
    n_recon_classes,
    n_mission_new_classes,
    acc_base: float,
    acc_recon: float,
    acc_new: float,
    experiment: ExperimentLogger,
    total_classes: int,
) -> ExperimentLogger:
    if acc_recon is None:
        acc_recon = np.nan
    fn = "eval_1_test.csv"
    cols = (
        "accuracy_set",
        "dataset",
        "mission_number",
        "client_id",
        "n_clients",
        "n_base_classes",
        "n_recon_classes",
        "n_mission_classes",
        "total_n_classes",
        "accuracy",
    )
    row = [
        "base",
        "test",
        mission_number,
        client_id,
        n_clients,
        n_base_classes,
        n_recon_classes,
        n_mission_new_classes,
        total_classes,
        acc_base,
    ]
    experiment.record_result(fn, cols, row)
    row = [
        "reconned",
        "test",
        mission_number,
        client_id,
        n_clients,
        n_base_classes,
        n_recon_classes,
        n_mission_new_classes,
        total_classes,
        acc_recon,
    ]
    experiment.record_result(fn, cols, row)
    row = [
        "mission_new",
        "test",
        mission_number,
        client_id,
        n_clients,
        n_base_classes,
        n_recon_classes,
        n_mission_new_classes,
        total_classes,
        acc_new,
    ]
    experiment.record_result(fn, cols, row)
    row = [
        "averaged_by_task_set",
        "test",
        mission_number,
        client_id,
        n_clients,
        n_base_classes,
        n_recon_classes,
        n_mission_new_classes,
        total_classes,
        np.nanmean([acc_base, acc_recon, acc_new]),
    ]
    experiment.record_result(fn, cols, row)
    return experiment


def record_eval_2(
    mission_number: int,
    n_clients: int,
    n_base_classes,
    n_field_classes,
    acc_base: float,
    acc_field: float,
    experiment: ExperimentLogger,
):
    fn = "eval_2_test.csv"
    cols = (
        "task_set",
        "dataset",
        "mission_number",
        "n_clients",
        "n_base_classes",
        "n_field_classes",
        "total_n_classes",
        "accuracy",
    )
    row = [
        "base",
        "test",
        mission_number,
        n_clients,
        n_base_classes,
        n_field_classes,
        n_base_classes + n_field_classes,
        acc_base,
    ]
    experiment.record_result(fn, cols, row)
    row = [
        "field",
        "test",
        mission_number,
        n_clients,
        n_base_classes,
        n_field_classes,
        n_base_classes + n_field_classes,
        acc_field,
    ]
    experiment.record_result(fn, cols, row)
    return experiment


class Server:
    """ Initialize the server """

    def __init__(
        self,
        num_clients,
        environment: MiniImagenet,
        model,
        train_base_model: bool = True,
        update_base_memory: bool = True,
        experiment: Optional[ExperimentLogger] = None,
        eval_clients_during_recon: bool = True,
    ):
        print("[Server] initialization")
        self.num_clients = num_clients
        self.environment = environment
        self.experiment = experiment
        self.eval_clients_during_recon = eval_clients_during_recon
        self.clients = []
        # Set of classes learned by clients not in the base training set:
        self.reconned_classes: Set = set()
        # bool to figure out if model is being updated
        self.updating = False
        for _ in range(num_clients):
            client_id = len(self.clients)
            client_name = "client" + str(client_id)
            client = Client(client_id, client_name, environment)
            self.clients.append(client)

        print(f"[Server] {num_clients} Clients initialized")
        # new classes returned by clients, client_id : list of class_ids for a single mission
        self.new_classes = {}
        # new models pushed by clients, client_id : model
        self.new_models = {}
        self.model = model

        if train_base_model:
            print("[Server] training base model on base train")
            self.model.train_base_model(self.environment.base_train)
        if update_base_memory:
            print("[Server] updating memory of base model on base training classes")
            self.model.update_base_memory(self.environment.base_train)

        print("[Server] testing base model on base test examples")
        acc = self.model.test_model(self.environment.base_test)
        print(f"[Server] Test accuracy: {round(acc, 2)}")
        if self.experiment is not None:
            self.experiment = record_eval_0(acc, self.experiment)

    def check_model_updating(self):
        """ Check if the model is being updated """
        return self.updating

    def get_model(self):
        """ Return updated model to client """
        # will need to handle this more closely later
        # currently should not be a problem
        model_updating = self.check_model_updating()
        if model_updating:
            raise ModelUpdatingException
        else:
            print("[Server] sending new model instance to client")
            return deepcopy(self.model)

    def add_client(self):
        """ Add new client to Server """
        client_id = len(self.clients)
        client_name = "client" + str(client_id)
        client = Client(client_id, client_name, self.environment)
        self.clients.append(client)
        self.num_clients += 1

    def merge_models(self):
        """ Merge the model obtained from clients """
        self.model.merge_models(self.new_models, self.new_classes)
        # update the classes
        for _, class_list in self.new_classes.items():
            self.reconned_classes.update(class_list)

        # re initialize
        self.new_classes = {}
        self.new_models = {}
        print("[Server] models merged")

    def test_model(self, mission_number: int, n_clients: int):
        """ Test model on all base classes and new classes """
        print(f"[Server] testing model on base test and new classes")
        new_labels = list(self.reconned_classes)

        sub_dataset = self.environment.get_mission_test_data(new_labels)
        # dataset = ConcatDataset([sub_dataset, self.environment.base_test])
        acc_sub = self.model.test_model(sub_dataset)
        print(
            f"[Server] Test accuracy on {len(new_labels)} new classes {new_labels}: {round(acc_sub, 2)}"
        )
        acc_base = self.model.test_model(self.environment.base_test)
        print(f"[Server] Test accuracy on base test classes: {round(acc_base, 2)}")

        if self.experiment is not None:
            self.experiment = record_eval_2(
                mission_number,
                n_clients,
                len(self.environment.base_labels),
                len(new_labels),
                acc_base,
                acc_sub,
                self.experiment,
            )

    def send_mission(
        self, mission_number: int, n_clients_in_mission: Optional[int] = None
    ):
        # randomly select some clients and send them to mission
        # how many clients will go to mission
        # n_clients_in_mission = random.sample(range(self.num_clients), 1)

        if n_clients_in_mission is None:
            n_clients_in_mission = self.num_clients
        # which clients will go to mission
        print(f"[Server] sending {n_clients_in_mission} clients to mission")
        samples = random.sample(range(self.num_clients), n_clients_in_mission)
        current_mission = []
        for sample in samples:
            current_mission.append(self.clients[sample])

        for client in current_mission:
            print(f"[Server] sending new model to client {client.id}")
            client.go_to_mission(self.get_model())
            if client.mission_data is None or client.mission_labels is None:
                # For now we assume we need both inputs and labels for on-client learning
                warnings.warn(
                    f"Client {client.id} received no new data on mission {mission_number}."
                )
                continue
            if self.eval_clients_during_recon:
                (
                    base_acc,
                    recon_acc,
                    new_classes_acc,
                ) = client.evalute_client_model_on_mission(
                    reconned_classes=self.reconned_classes
                )
                if self.experiment is not None:
                    self.experiment = record_eval_1(
                        mission_number,
                        client.id,
                        n_clients_in_mission,
                        len(self.environment.base_labels),
                        len(self.reconned_classes),
                        len(client.mission_classes),
                        base_acc,
                        recon_acc,
                        new_classes_acc,
                        self.experiment,
                        total_classes=len(
                            set(
                                self.environment.base_labels
                                + list(self.reconned_classes)
                                + client.mission_classes
                            )
                        ),
                    )
            new_model = client.give_model_to_server()
            self.new_models[client.id] = new_model
            new_classes = client.give_new_classes_to_server()
            self.new_classes[client.id] = new_classes
            client.clean()

        self.merge_models()
        # Eval 2 diamond in fed recon diagram:
        self.test_model(mission_number, n_clients_in_mission)

    def save_results(self):
        """Serialize the experiment results to disk."""
        self.experiment.serialize_results()
