from typing import Tuple, List, Dict, Set, Union, Optional

from torch.utils.data.dataset import Dataset
from torchvision import transforms
import random
import os
from PIL import Image
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset

TOTAL_TRAINING_EXAMPLES_PER_CLASS = 500
TOTAL_TEST_EXAMPLES_PER_CLASS = 100
TOTAL_TRAINING_EXAMPLES_PER_CLASS_BASE = 450
TOTAL_TRAINING_EXAMPLES_PER_CLASS_FIELD = 500
VALIDATION_EXAMPLES_PER_BASE_CLASS = 50
HOLD_OUT_VALIDATION_CLASSES = 10
HOLD_OUT_TRAINING_EXAMPLES_PER_CLASS = 500


class BaseTrain(Dataset):
    def __init__(self, images_path, data, transform):
        super().__init__()
        self.images_path = images_path
        self.data = data
        self.transform = transform

        self._image_cache = {}

        self.total_examples_per_class = 500

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        path = self.images_path + "/" + img_name

        try:
            img = self._image_cache[path]
        except KeyError:
            img = Image.open(path)
            img = self.transform(img)
            img = img.numpy()
            self._image_cache[path] = img
        return torch.from_numpy(img), label


class BaseVal(Dataset):
    def __init__(self, images_path, data, transform):
        super().__init__()
        self.images_path = images_path
        self.data = data
        self.transform = transform

        self._image_cache = {}

        self.total_examples_per_class = 500

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        path = self.images_path + "/" + img_name

        try:
            img = self._image_cache[path]
        except KeyError:
            img = Image.open(path)
            img = self.transform(img)
            img = img.numpy()
            self._image_cache[path] = img
        return torch.from_numpy(img), label


class BaseTest(Dataset):
    def __init__(self, images_path, data, transform):
        super().__init__()
        self.images_path = images_path
        self.data = data
        self.transform = transform

        self._image_cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        path = self.images_path + "/" + img_name

        try:
            img = self._image_cache[path]
        except KeyError:
            img = Image.open(path)
            img = self.transform(img)
            img = img.numpy()
            self._image_cache[path] = img

        return torch.from_numpy(img), label


class HoldoutTrain(Dataset):
    def __init__(self, images_path, data, transform):
        super().__init__()
        self.images_path = images_path
        self.data = data
        self.transform = transform

        self._image_cache = {}

        self.total_examples_per_class = 500

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        path = self.images_path + "/" + img_name

        try:
            img = self._image_cache[path]
        except KeyError:
            img = Image.open(path)
            img = self.transform(img)
            img = img.numpy()
            self._image_cache[path] = img
        return torch.from_numpy(img), label


class HoldoutVal(Dataset):
    def __init__(self, images_path, data, transform):
        super().__init__()
        self.images_path = images_path
        self.data = data
        self.transform = transform

        self._image_cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        path = self.images_path + "/" + img_name

        try:
            img = self._image_cache[path]
        except KeyError:
            img = Image.open(path)
            img = self.transform(img)
            img = img.numpy()
            self._image_cache[path] = img

        return torch.from_numpy(img), label


class FieldTrain(Dataset):
    def __init__(self, images_path, data, transform):
        super().__init__()
        self.images_path = images_path
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img = Image.open(self.images_path + "/" + img_name)  # .convert('RGB')
        return self.transform(img), label


class FieldTest(Dataset):
    def __init__(self, images_path, data, transform):
        super().__init__()
        self.images_path = images_path
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img = Image.open(self.images_path + "/" + img_name)  # .convert('RGB')
        return self.transform(img), label


class PerClassDataset(Dataset):
    def __init__(self, images_path, data, transform):
        super().__init__()
        self.images_path = images_path
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img = Image.open(self.images_path + "/" + img_name)  # .convert('RGB')
        return self.transform(img), label


class SubDataset(Dataset):
    def __init__(self, dataset, class_list):
        super().__init__()
        self.dataset = dataset
        self.class_list = class_list
        self.sub_indices = []

        for idx in range(len(self.dataset)):
            label = self.dataset[idx][1]
            if label in class_list:
                self.sub_indices.append(idx)

    def __len__(self):
        return len(self.sub_indices)

    def __getitem__(self, index):
        return self.dataset[self.sub_indices[index]]


def get_base_field_classes(
    hold_out_bool: bool = False,
) -> Union[
    Tuple[List[str], List[str], List[str], List[str], Dict[str, str], Dict[str, str]],
    Tuple[List[str], List[str], List[str], Dict[str, str], Dict[str, str]],
]:
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "classes.txt")
    id_to_name = {}
    name_to_id = {}
    class_ids = []
    with open(filename, "r") as read_file:
        for line in read_file:
            class_id, class_name = line.strip().split(" ")
            class_ids.append(class_id)
            id_to_name[class_id] = class_name
            name_to_id[class_name] = class_id

    temp_classes = class_ids[: int(len(class_ids) / 2)]
    if hold_out_bool:
        base_classes = temp_classes[: len(temp_classes) - HOLD_OUT_VALIDATION_CLASSES]
        hold_out_classes = temp_classes[-HOLD_OUT_VALIDATION_CLASSES:]
        field_classes = class_ids[int(len(class_ids) / 2) :]
        return (
            base_classes,
            hold_out_classes,
            field_classes,
            class_ids,
            id_to_name,
            name_to_id,
        )
    else:
        base_classes = temp_classes
        field_classes = class_ids[int(len(class_ids) / 2) :]
        return base_classes, field_classes, class_ids, id_to_name, name_to_id


def get_class_id_images_map(images_path: str):
    dataset = {}
    image_names = [
        image_name
        for image_name in os.listdir(images_path + "/")
        if image_name.endswith(".jpg")
    ]
    # Sort for reproducibility:
    image_names = sorted(image_names)
    for img_name in image_names:
        class_id = img_name[:9]
        if class_id not in dataset.keys():
            dataset[class_id] = [img_name]
        else:
            dataset[class_id].append(img_name)
    return dataset


def get_base_field_splits(
    dataset: Dict[str, List[str]],
    base_classes: List[str],
    field_classes: List[str],
    class_ids: List[str],
    training_examples_per_class_train: int = TOTAL_TRAINING_EXAMPLES_PER_CLASS_BASE,
    training_examples_per_class_field: int = TOTAL_TRAINING_EXAMPLES_PER_CLASS_FIELD,
    hold_out_classes: Optional[List[str]] = None,
    base_val_bool: bool = False,
):
    """
    Builds the train test splits for the base and field classes
    :return:
    """
    if base_val_bool:
        base = {"train": [], "val": [], "test": []}
    else:
        base = {"train": [], "test": []}

    if hold_out_classes is not None:
        hold_out = {"train": [], "val": []}
    else:
        hold_out = None

    field = {"train": [], "test": []}
    base_labels, hold_out_labels, field_labels = set(), set(), set()

    # create dataset for each class for field train and field test
    # so that when we want to load random batches we can get them directly instead of
    # creating subDataset which has to loop through each index
    field_train_per_class = {}
    field_test_per_class = {}
    for class_id in field_classes:
        field_train_per_class[class_id] = []
        field_test_per_class[class_id] = []

    if base_val_bool:
        first_split = training_examples_per_class_train
        second_split = (
            training_examples_per_class_train + VALIDATION_EXAMPLES_PER_BASE_CLASS
        )
    else:
        first_split = (
            training_examples_per_class_train + VALIDATION_EXAMPLES_PER_BASE_CLASS
        )

    for class_id, img_names in dataset.items():
        if class_id in base_classes:
            label = class_ids.index(class_id)
            base_labels.add(label)

            for img_name in img_names[:first_split]:
                base["train"].append((img_name, label))

            if base_val_bool:
                for img_name in img_names[first_split:second_split]:
                    base["val"].append((img_name, label))

                for img_name in img_names[second_split:]:
                    base["test"].append((img_name, label))
            else:
                for img_name in img_names[first_split:]:
                    base["test"].append((img_name, label))

        elif class_id in field_classes:
            label = class_ids.index(class_id)
            field_labels.add(label)
            for img_name in img_names[:training_examples_per_class_field]:
                field["train"].append((img_name, label))
                field_train_per_class[class_id].append((img_name, label))
            for img_name in img_names[training_examples_per_class_field:]:
                field["test"].append((img_name, label))
                field_test_per_class[class_id].append((img_name, label))

        elif class_id in hold_out_classes:
            label = class_ids.index(class_id)
            hold_out_labels.add(label)
            for img_name in img_names[:HOLD_OUT_TRAINING_EXAMPLES_PER_CLASS]:
                hold_out["train"].append((img_name, label))
            for img_name in img_names[HOLD_OUT_TRAINING_EXAMPLES_PER_CLASS:]:
                hold_out["val"].append((img_name, label))

        else:
            raise ValueError("Unable to find class_id")

    if hold_out_classes is not None:
        return (
            base,
            list(base_labels),
            hold_out,
            list(hold_out_labels),
            field,
            list(field_labels),
            field_train_per_class,
            field_test_per_class,
        )
    else:
        return base, field, field_train_per_class, field_test_per_class


class MiniImagenet:
    """
    images_path example: "data/mini-imagenet/images"
    """

    def __init__(
        self,
        images_path,
        training_examples_per_class_per_mission: int = 30,
        sample_random_number_of_classes: bool = False,
        n_classes_per_mission: int = 5,
        sample_without_replacement: bool = True,
        hold_out_bool=False,
        base_val_bool=False,
    ):
        self.images_path = images_path
        self.training_examples_per_class_per_mission = (
            training_examples_per_class_per_mission
        )
        self.sample_random_number_of_classes = sample_random_number_of_classes
        self.n_classes_per_mission = n_classes_per_mission
        self.sample_without_replacement = sample_without_replacement

        self.previously_sampled_field_train_images: Set[str] = set()

        self.transform = transforms.Compose(
            [
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if hold_out_bool:
            (
                self.base_classes,
                self.hold_out_classes,
                self.field_classes,
                self.class_ids,
                self.id_to_name,
                self.name_to_id,
            ) = get_base_field_classes(hold_out_bool)
        else:
            (
                self.base_classes,
                self.field_classes,
                self.class_ids,
                self.id_to_name,
                self.name_to_id,
            ) = get_base_field_classes(hold_out_bool)

        total_classes = len(self.id_to_name)
        if not hold_out_bool:
            self.labels: List[int] = list(range(total_classes))
            self.id_to_label = {
                key: val for key, val in zip(self.id_to_name.keys(), self.labels)
            }
            self.base_labels: List[int] = self.labels[: int(total_classes / 2)]
            self.field_labels: List[int] = self.labels[int(total_classes / 2) :]

        for item in self.field_classes:
            assert item not in self.base_classes
            if hold_out_bool:
                assert item not in self.hold_out_classes

        if hold_out_bool:
            for item in self.hold_out_classes:
                assert item not in self.base_classes
                assert item not in self.field_classes

        for item in self.base_classes:
            assert item not in self.field_classes
            if hold_out_bool:
                assert item not in self.hold_out_classes

        dataset = get_class_id_images_map(images_path)
        for key, val in dataset.items():
            assert all([key in fn for fn in val])
        assert len(dataset) == total_classes

        if hold_out_bool:
            (
                self.base,
                self.base_labels,
                self.hold_out,
                self.hold_out_labels,
                self.field,
                self.field_labels,
                self.field_train_per_class,
                self.field_test_per_class,
            ) = get_base_field_splits(
                dataset,
                self.base_classes,
                self.field_classes,
                self.class_ids,
                TOTAL_TRAINING_EXAMPLES_PER_CLASS_BASE,
                TOTAL_TRAINING_EXAMPLES_PER_CLASS_FIELD,
                hold_out_classes=self.hold_out_classes,
                base_val_bool=base_val_bool,
            )
        else:
            (
                self.base,
                self.field,
                self.field_train_per_class,
                self.field_test_per_class,
            ) = get_base_field_splits(
                dataset,
                self.base_classes,
                self.field_classes,
                self.class_ids,
                TOTAL_TRAINING_EXAMPLES_PER_CLASS_BASE,
                TOTAL_TRAINING_EXAMPLES_PER_CLASS_FIELD,
                base_val_bool=base_val_bool,
            )

        assert set(self.base["train"]).isdisjoint(set(self.base["test"]))
        assert set(self.field["train"]).isdisjoint(set(self.field["test"]))

        self.base_train = BaseTrain(
            self.images_path, self.base["train"], self.transform
        )
        if base_val_bool:
            self.base_val = BaseVal(self.images_path, self.base["val"], self.transform)
        else:
            self.base_val = None

        self.base_test = BaseTest(self.images_path, self.base["test"], self.transform)

        if hold_out_bool:
            self.hold_out_train = HoldoutTrain(
                self.images_path, self.hold_out["train"], self.transform
            )
            self.hold_out_val = HoldoutVal(
                self.images_path, self.hold_out["val"], self.transform
            )
        else:
            self.hold_out_train = None
            self.hold_out_val = None

        self.field_train = FieldTrain(
            self.images_path, self.field["train"], self.transform
        )
        self.field_test = FieldTest(
            self.images_path, self.field["test"], self.transform
        )
        self.field_train_per_class_dataset = {}
        self.field_test_per_class_dataset = {}

        for class_id in self.field_classes:
            self.field_train_per_class_dataset[class_id] = PerClassDataset(
                self.images_path, self.field_train_per_class[class_id], self.transform
            )

            self.field_test_per_class_dataset[class_id] = PerClassDataset(
                self.images_path, self.field_test_per_class[class_id], self.transform
            )
        print("[Environment] initialized")

        self.class_sampling_counts = (
            {}
        )  # Map from class to number of examples sampled for that class
        self.class_unique_samples = {
            class_id: {} for class_id in self.field_labels
        }  # Map from class to image path to label

    @property
    def n_total_field_training_examples(self):
        return len(self.field["train"])

    def get_class_name_from_id(self, class_id):
        if class_id in self.id_to_name.keys():
            return self.id_to_name[class_id]
        else:
            raise KeyError(f"Class {class_id} not found")

    def get_class_id_from_name(self, class_name):
        if class_name in self.name_to_id.keys():
            return self.name_to_id[class_name]
        else:
            raise KeyError(f"Class {class_name} not found")

    def get_class_id_from_label(self, label):
        return self.class_ids[label]

    def get_class_name_from_label(self, label):
        class_id = self.get_class_id_from_label(label)
        return self.get_class_name_from_id(class_id)

    def get_label_from_id(self, class_id):
        if class_id in self.class_ids:
            return self.class_ids.index(class_id)
        else:
            raise KeyError(f"Class {class_id} not found")

    def get_datasets(self):
        return (
            self.base_train,
            self.base_val,
            self.base_test,
            self.hold_out_train,
            self.hold_out_val,
            self.field_train,
            self.field_test,
        )

    def get_base_data_test_tensor(self, selected_classes):
        test_examples_per_class = TOTAL_TEST_EXAMPLES_PER_CLASS
        print(f"[Environment] getting test base data for class: {selected_classes}")
        assert all([x in self.base_labels for x in selected_classes])
        total_examples = test_examples_per_class * len(selected_classes)

        data = torch.zeros(total_examples, 3, 84, 84)
        labels = torch.tensor([0] * total_examples)

        for idx, class_id in enumerate(selected_classes):
            offset = idx * test_examples_per_class
            counter = 0
            for img_path, lab in self.base["test"]:
                if counter == test_examples_per_class:
                    break
                else:
                    if lab == class_id:
                        img_path = self.images_path + "/" + img_path
                        data[offset + counter] = self.transform(
                            Image.open(img_path)  # .convert("RGB")
                        )
                        labels[offset + counter] = class_id
                        counter += 1
                    else:
                        continue

        return data, labels

    def _get_field_training_examples(self):
        field_training_examples = self.field["train"][:]  # clone the list
        random.shuffle(field_training_examples)

        return field_training_examples

    def _sample_classes_for_mission(self) -> List[int]:
        if self.sample_random_number_of_classes:
            n = random.sample(range(1, 5), 1)[0]
        else:
            n = self.n_classes_per_mission
        if self.sample_without_replacement:
            # Filter to classes that have examples that haven't been seen yet:
            field_labels_with_examples_remaining = [
                class_id
                for class_id in self.class_unique_samples.keys()
                if len(self.class_unique_samples[class_id])
                < TOTAL_TRAINING_EXAMPLES_PER_CLASS_FIELD
            ]
            n = (
                len(field_labels_with_examples_remaining)
                if len(field_labels_with_examples_remaining) < n
                else n
            )
            if n > 0:
                selected_classes: List[int] = random.sample(
                    field_labels_with_examples_remaining, n
                )
            else:
                selected_classes = []
        else:
            selected_classes: List[int] = random.sample(self.field_labels, n)
        return selected_classes

    def get_mission_data_train_tensor(
        self, verbose = False
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns a sample of mission data from field classes for training clients.
        :return: tuple of torch.Tensor images, torch.Tensor labels, and list of int of classes that were attempted to sample
            Note that the  len(set(labels)) may not equal n_classes_per_mission if self.sample_without_replacement is True due to already having sampled all examples for a class in selected_classes
        """
        selected_classes = self._sample_classes_for_mission()
        print(f"[Environment] getting train mission data for class: {selected_classes}")
        assert all([x in self.field_labels for x in selected_classes])

        examples_per_class = self.training_examples_per_class_per_mission

        images: List[torch.Tensor] = []
        labels: List[int] = []

        for idx, class_id in enumerate(selected_classes):
            counter = 0
            field_training_examples = self._get_field_training_examples()
            for img_path, lab in field_training_examples:
                if self.sample_without_replacement:
                    if img_path in self.previously_sampled_field_train_images:
                        continue
                if counter == examples_per_class:
                    break
                if lab == class_id:
                    self.previously_sampled_field_train_images.add(img_path)
                    img_path = os.path.join(self.images_path, img_path)

                    images.append(self.transform(Image.open(img_path)))
                    labels.append(class_id)
                    counter += 1

                    try:
                        self.class_sampling_counts[class_id] += 1
                        self.class_unique_samples[class_id][img_path] = lab
                    except KeyError:
                        self.class_sampling_counts[class_id] = 1
                        self.class_unique_samples[class_id] = {img_path: lab}

        if verbose:
            print("Mini-imagenet class sampling counts: ", self.class_sampling_counts)
            print(
                "Mini-imagenet unique samples: ",
                {key: len(val) for key, val in self.class_unique_samples.items()},
            )

        if len(images) > 0 and len(labels) > 0:
            assert len(images) == len(labels)
            images: torch.Tensor = torch.stack(images, dim=0)
            labels: torch.Tensor = torch.tensor(labels)
            assert len(images.shape) == 4
            assert len(labels.shape) == 1
        else:
            images = None
            labels = None

        return images, labels

    def get_mission_data_test_tensor(self, selected_classes):
        test_examples_per_class = TOTAL_TEST_EXAMPLES_PER_CLASS
        print(f"[Environment] getting test mission data for class: {selected_classes}")
        assert all([x in self.field_labels for x in selected_classes])
        total_examples = test_examples_per_class * len(selected_classes)

        data = torch.zeros(total_examples, 3, 84, 84)
        labels = torch.tensor([0] * total_examples)

        for idx, class_id in enumerate(selected_classes):
            offset = idx * test_examples_per_class
            counter = 0
            for img_path, lab in self.field["test"]:
                if counter == test_examples_per_class:
                    break
                else:
                    if lab == class_id:
                        img_path = self.images_path + "/" + img_path
                        data[offset + counter] = self.transform(Image.open(img_path))
                        labels[offset + counter] = class_id
                        counter += 1
                    else:
                        continue

        return data, labels

    def get_mission_train_data(self):
        # FIXME: we should only have one of these functions, not 2.
        #   delete this function in favor of get_mission_data_train_tensor
        raise NotImplementedError("function get_mission_train_data is deprecated")
        print("[Environment] getting mission train data")
        examples_per_class = 128
        class_list = random.sample(self.field_classes, 2)
        field_labels = []
        data = None
        labels = []
        for class_id in class_list:
            field_labels.append(self.get_label_from_id(class_id))
            train_loader = DataLoader(
                self.field_train_per_class_dataset[class_id],
                batch_size=examples_per_class,
            )
            iterator = iter(train_loader)
            data_temp, labels_temp = next(iterator)
            if data is None:
                data = data_temp
            else:
                data = torch.cat((data, data_temp), dim=0)

            labels.extend(labels_temp)

        labels = torch.tensor(labels)
        return data, labels, field_labels

    def get_mission_test_data(self, labels_list):
        print("[Environment] getting mission test data")
        # class list consists of classes from meta test and field test
        class_list = []
        for label in labels_list:
            class_list.append(self.get_class_id_from_label(label))

        dataset = self.field_test_per_class_dataset[class_list[0]]
        for class_id in class_list[1:]:
            dataset = ConcatDataset(
                [dataset, self.field_test_per_class_dataset[class_id]]
            )

        return dataset
