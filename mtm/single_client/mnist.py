import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch


class SplitMNIST(Dataset):
    def __init__(self, dataset, sub_labels):
        super().__init__()
        self.dataset = dataset
        self.sub_labels = sub_labels
        self.sub_indices = []

        for idx in range(len(dataset)):
            if hasattr(dataset, "targets"):
                label = dataset.targets[idx]
            else:
                label = self.dataset[idx][1]
            if label in sub_labels:
                self.sub_indices.append(idx)

    def __len__(self):
        return len(self.sub_indices)

    def __getitem__(self, index):
        return self.dataset[self.sub_indices[index]]


def load_mnist_dataset(data_dir, num_tasks):
    classes_per_task = int(np.floor(10 / num_tasks))
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(
        root=data_dir, train=True, download=False, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=False, transform=transform
    )

    labels_per_task = [
        list(np.array(range(classes_per_task)) + classes_per_task * task_id)
        for task_id in range(num_tasks)
    ]

    train_dataset = []
    # test_dataset = []
    for labels in labels_per_task:
        train_dataset.append(SplitMNIST(train, labels))
        # test_dataset.append(SplitMNIST(test, labels))

    return train_dataset, test_dataset, labels_per_task


def load_mnist_dataset_joint(data_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=False, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=False, transform=transform
    )
    return train_dataset, test_dataset


class ExemplarDataset(Dataset):
    """
    Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).
    The images at the i-th entry of [exemplar_sets] belong to class [i]
    """

    def __init__(self, exemplar_sets):
        super().__init__()
        self.exemplar_sets = exemplar_sets

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        return_class_id, class_id, exemplar_id = None, None, None
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                return_class_id = class_id
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
        return image, return_class_id
