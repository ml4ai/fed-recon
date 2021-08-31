"""
Code for training models and running single client incremental few-shot learning benchmarks as presented in Ren et al. 2019
"""
import os
import random
from typing import Callable, Optional, Union, Set, Dict, List, Tuple

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from mtm.fed_recon.mini_imagenet import (
    get_base_field_classes,
    get_class_id_images_map,
    get_base_field_splits,
    TOTAL_TRAINING_EXAMPLES_PER_CLASS,
)
from mtm.util.util import ncr

VALID_META_SPLITS = {"train", "val", "test"}
SPLIT_FILENAMES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}
IMAGES_SUBDIR = "images"
IMAGE_RESIZE_HW = (84, 84)


def get_unique_classes(path: str):
    df = pd.read_csv(path)
    assert "label" in df.columns.values.tolist()
    return df["label"].unique()


def _setup_file_paths_df(
    root_dir: str, meta_splits: Set[str], fed_recon: bool = False
) -> Dict[str, pd.DataFrame]:
    """Setup the filepaths dataframe containing three columns: filename, label, path"""
    if fed_recon:
        return _setup_file_paths_df_fed_recon(root_dir, meta_splits)
    split_dfs = {
        split: pd.read_csv(os.path.join(root_dir, SPLIT_FILENAMES[split]))
        for split in meta_splits
    }
    for split, df in split_dfs.items():
        df["path"] = df["filename"].apply(
            lambda x: os.path.join(root_dir, IMAGES_SUBDIR, x)
        )
        split_dfs[split] = df
    return split_dfs


def _setup_file_paths_df_fed_recon(
    root_dir: str, meta_splits: Set[str], filter_out_test_images_per_class: bool = True
) -> Dict[str, pd.DataFrame]:
    if "val" in meta_splits:
        raise NotImplementedError(
            "TODO: need to implement val support (and train+val?) support for fedrecon mini-imagenet"
        )
    (
        base_classes,
        field_classes,
        class_ids,
        id_to_name,
        name_to_id,
    ) = get_base_field_classes()
    class_id_image_map = get_class_id_images_map(os.path.join(root_dir, IMAGES_SUBDIR))
    base_splits, field_splits, _, _ = get_base_field_splits(
        class_id_image_map, base_classes, field_classes, class_ids, base_val_bool=False
    )

    base_test_images, _ = zip(*base_splits["test"])
    field_test_images, _ = zip(*field_splits["test"])
    all_test_images = base_test_images + field_test_images

    dfs = []
    for _, basename in SPLIT_FILENAMES.items():
        dfs.append(pd.read_csv(os.path.join(root_dir, basename)))
    df = pd.concat(dfs)

    if filter_out_test_images_per_class:
        # Skip base and field test examples:
        df = df[~df["filename"].isin(all_test_images)]

    split_dfs = {}
    if "train" in meta_splits:
        split_dfs["train"] = df[
            df["filename"].str.contains("|".join(base_classes), regex=True)
        ]
    if "test" in meta_splits:
        split_dfs["test"] = df[
            df["filename"].str.contains("|".join(field_classes), regex=True)
        ]
    if "val" in meta_splits:
        raise NotImplementedError(
            "TODO: need to implement val support (and train+val?) support for fedrecon mini-imagenet"
        )

    if filter_out_test_images_per_class:
        should_be = 25000
    else:
        should_be = 30000
    sum_should_be = should_be * 2
    if meta_splits == {"train"}:
        assert sum([len(x) for x in split_dfs.values()]) == should_be
    elif meta_splits == {"test"}:
        assert sum([len(x) for x in split_dfs.values()]) == should_be
    elif meta_splits == {"train", "test"}:
        assert sum([len(x) for x in split_dfs.values()]) == sum_should_be
    elif meta_splits == {"train", "val", "test"}:
        raise NotImplementedError(
            "TODO: need to implement val support (and train+val?) support for fedrecon mini-imagenet"
        )
    else:
        raise ValueError(
            f"meta_splits should be in {VALID_META_SPLITS} but is {meta_splits}"
        )

    for split, df in split_dfs.items():
        df["path"] = df["filename"].apply(
            lambda x: os.path.join(root_dir, IMAGES_SUBDIR, x)
        )
        split_dfs[split] = df
    return split_dfs


def _setup_split_classes(split_dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """build a map from split to classes in that split"""
    split_classes = {}
    for split, df in split_dfs.items():
        assert "label" in df.columns.values.tolist()
        classes = sorted(list(df["label"].unique()))
        split_classes[split] = classes
    classes = [x for x in split_classes.values()]
    for i in range(len(classes)):
        s1 = classes[i]
        assert len(s1) == len(set(s1))
        for j in range(i + 1, len(classes)):
            s2 = classes[j]
            assert set(s1).isdisjoint(set(s2))
    return split_classes


def _setup_file_paths_dict(meta_splits: Set[str], split_dfs: Dict[str, pd.DataFrame]):
    split_class_paths = {split: {} for split in meta_splits}
    for split, df in split_dfs.items():
        for i, row in df.iterrows():
            try:
                split_class_paths[split][row["label"]].append(row["path"])
            except KeyError:
                split_class_paths[split][row["label"]] = [row["path"]]
    return split_class_paths


class MiniImagenet:
    def __init__(
        self,
        root_dir: str,
        n_classes_per_task: Optional[int] = 5,
        k_shots_per_class: Optional[int] = 1,
        test_shots_per_class: Optional[int] = 1,
        meta_split: Union[str, Set[str]] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_resize_hw: Tuple[int, int] = IMAGE_RESIZE_HW,
        fed_recon: bool = False,
    ):
        assert os.path.isdir(root_dir)
        assert isinstance(n_classes_per_task, int)
        assert isinstance(k_shots_per_class, int)
        assert isinstance(image_resize_hw, tuple)
        assert len(image_resize_hw) == 2
        if isinstance(meta_split, str):
            assert meta_split in VALID_META_SPLITS
            self.meta_splits = {meta_split}
        elif isinstance(meta_split, set):
            assert all([x in VALID_META_SPLITS for x in meta_split])
        else:
            raise ValueError(
                "Unsupported type {}, should be str or set.".format(type(meta_split))
            )

        self.root_dir = root_dir
        self.n_classes_per_task = n_classes_per_task
        self.k_shots_per_class = k_shots_per_class
        self.test_shots_per_class = test_shots_per_class
        self.split_dfs: Dict[str, pd.DataFrame] = _setup_file_paths_df(
            self.root_dir, self.meta_splits, fed_recon=fed_recon
        )
        self.split_classes: Dict[str, List[str]] = _setup_split_classes(self.split_dfs)
        # map from meta split to class name to list of paths:
        self.split_class_paths: Dict[
            str, Dict[str, List[str]]
        ] = _setup_file_paths_dict(self.meta_splits, self.split_dfs)
        self.total_n_classes = sum([len(x) for x in self.split_classes.values()])

        self.image_resize_hw = image_resize_hw
        if transform is None:
            transform = Compose(
                [
                    Resize(image_resize_hw),
                    ToTensor(),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        if target_transform is None:
            target_transform = torch.tensor
        self.transform = transform
        self.target_transform = target_transform

        if self.total_n_classes is not None and self.n_classes_per_task is not None:
            self.len = ncr(self.total_n_classes, self.n_classes_per_task)
        else:
            self.len = None

        self._image_cache = {}

    def sample_meta_batch(
        self,
        batch_size: int = 1,
        meta_split: str = "train",
        n_classes_per_task: Optional[int] = None,
        k_shots_per_class: Optional[int] = None,
        test_shots_per_class: Optional[int] = None,
        return_n_k_along_same_axis: bool = True,
        sample_k_value: bool = False,
    ):
        """
        Returns data for a classification task in a dictionary
        mapping string to tuple of mini-ImageNet {"{train, test}": (images, labels)}.
        If return_n_k_along_same_axis, data is in shape:
            ([b, n, k, channels, rows, cols], [b, n, k])
        else:
            ([b, n * k, channels, rows, cols], [b, n * k])
        Each element in the label tensor is the integer index representing the class
        where:
            b is meta-batch size
            n is the number of ways/classes
            k is the number of shots per class
            channels is the number of channels in the image
            rows is the number of rows in the image
            cols is the number of columns in the image

        NOTE: Examples for each class are not shuffled.

        :param batch_size: int of number of meta-tasks to sample.
        :param meta_split: str in {"train", "val", "test"} determining meta-split to sample from
        :param n_classes_per_task: optional number of classes to sample from when constructing a class. If None, will reference self.n_classes_per_task
        :param k_shots_per_class: optional number of training shots to sample for each class. If None, will reference self.k_shots_per_class
        :param test_shots_per_class: optional number of test shots to sample for each class. If None, will reference self.test_shots_per_class
        :return: dictionary mapping string to tuple of mini-ImageNet {"{train, test}": (images, labels)} in shape:
            ([b, n, k, channels, rows, cols], [b, n, k])
        """
        n_classes_per_task = (
            n_classes_per_task
            if n_classes_per_task is not None
            else self.n_classes_per_task
        )
        k_shots_per_class = (
            k_shots_per_class
            if k_shots_per_class is not None
            else self.k_shots_per_class
        )
        test_shots_per_class = (
            test_shots_per_class
            if test_shots_per_class is not None
            else self.test_shots_per_class
        )
        assert (
            n_classes_per_task is not None
            and k_shots_per_class is not None
            and test_shots_per_class is not None
        ), "n classes, k shots, and test shots must be specified in either class init or method call but found {}. {}, {}".format(
            n_classes_per_task, k_shots_per_class, test_shots_per_class
        )
        assert meta_split in self.meta_splits, "{} not in {}".format(
            meta_split, self.meta_splits
        )

        if sample_k_value:
            k_shots_per_class = random.randint(2, 50)
            test_shots_per_class = k_shots_per_class
        # sample k_shots_per_class for each class
        meta_batch_images = []
        meta_batch_labels = []
        for i in range(batch_size):
            # sample n_classes_per_task classes:
            classes = np.random.choice(
                self.split_classes[meta_split], n_classes_per_task, replace=False
            )
            task_images = []
            task_labels = []
            for j, class_name in enumerate(classes):
                paths_for_class = np.random.choice(
                    self.split_class_paths[meta_split][class_name],
                    k_shots_per_class + test_shots_per_class,
                    replace=False,
                )
                assert len(paths_for_class) == len(set(paths_for_class))
                images_for_class = []
                labels_for_class = []
                for path in paths_for_class:
                    try:
                        img = self._image_cache[path]
                    except KeyError:
                        img = Image.open(path)
                        img = self.transform(img)
                        img = img.numpy()
                        self._image_cache[path] = img
                    # TODO: split train and test set sampling, add image augmentation to training examples, and move img.numpy below that
                    images_for_class.append(img)
                    labels_for_class.append(j)
                task_images.append(images_for_class)
                task_labels.append(labels_for_class)
            meta_batch_images.append(task_images)
            meta_batch_labels.append(task_labels)

        meta_batch_images = np.array(meta_batch_images, dtype=np.float32)
        meta_batch_labels = np.array(meta_batch_labels, dtype=np.int64)
        train_images = meta_batch_images[:, :, :k_shots_per_class, :, :, :]
        test_images = meta_batch_images[:, :, k_shots_per_class:, :, :, :]
        train_labels = meta_batch_labels[:, :, :k_shots_per_class]
        test_labels = meta_batch_labels[:, :, k_shots_per_class:]

        train_images = torch.from_numpy(train_images)  # -> [b, n, k, ch, row, col]
        train_labels = torch.from_numpy(train_labels)  # -> [b, n, k]
        test_images = torch.from_numpy(test_images)  # -> [b, n, k, ch, row, col]
        test_labels = torch.from_numpy(test_labels)  # -> [b, n, k]

        if return_n_k_along_same_axis:
            train_images = train_images.reshape(
                [
                    batch_size,
                    n_classes_per_task * k_shots_per_class,
                    3,
                    *self.image_resize_hw,
                ]
            )
            train_labels = train_labels.reshape(
                [batch_size, n_classes_per_task * k_shots_per_class]
            )
            test_images = test_images.reshape(
                [
                    batch_size,
                    n_classes_per_task * test_shots_per_class,
                    3,
                    *self.image_resize_hw,
                ]
            )
            test_labels = test_labels.reshape(
                [batch_size, n_classes_per_task * test_shots_per_class]
            )

        # TODO add flag to shuffle
        # images_labels = list(zip(task_images, task_labels))
        # random.shuffle(images_labels)
        # task_images, task_labels = zip(*images_labels)

        batch = {
            "train": (train_images, train_labels),
            "test": (test_images, test_labels),
        }
        return batch

    def __len__(self):
        return self.len

    def __getitem__(self, index: Tuple[int]):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        meta_batch = self.sample_meta_batch()
        return meta_batch
