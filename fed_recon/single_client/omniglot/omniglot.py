"""
Sampling of omniglot examples.

Data is expected to exist in `root_dir` as:
root_dir/
    images_background/
        {train alphabet 1}/
            0709_01.png
            ...
        ...
        {train alphabet n}
    images_evaluation/
        {test alphabet 1}/
            0965_01.png
            ...
        ...
        {test alphabet m}/

Data was downloaded from: https://github.com/brendenlake/omniglot
"""
import os
from typing import Optional, Callable, Tuple, List, Dict

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

VALID_META_SPLITS = {"train": "images_background", "test": "images_evaluation"}
IMAGE_RESIZE_HW = (28, 28)


class Omniglot:
    def __init__(
        self,
        root_dir: str,
        n_classes_per_task: Optional[int] = 5,
        k_shots_per_class: Optional[int] = 1,
        test_shots_per_class: Optional[int] = 1,
        meta_split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_resize_hw: Tuple[int, int] = IMAGE_RESIZE_HW,
        fed_recon: bool = False,
    ):
        if fed_recon:
            raise NotImplementedError(
                "Federated reconnaissance not implemented for omniglot."
            )
        assert os.path.isdir(root_dir)
        assert isinstance(n_classes_per_task, int)
        assert isinstance(k_shots_per_class, int)
        assert isinstance(image_resize_hw, tuple)
        assert len(image_resize_hw) == 2
        assert isinstance(meta_split, str)
        assert meta_split in VALID_META_SPLITS

        self.root_dir = root_dir
        self.n_classes_per_task = n_classes_per_task
        self.k_shots_per_class = k_shots_per_class
        self.test_shots_per_class = test_shots_per_class
        self.meta_split = meta_split

        self.all_classes, self.class_paths = self._setup_classes()

        self.image_resize_hw = image_resize_hw
        if transform is None:
            transform = Compose([Resize(image_resize_hw), ToTensor()])
        if target_transform is None:
            target_transform = torch.tensor
        self.transform = transform
        self.target_transform = target_transform

        self._image_cache = {}

    def _setup_classes(self) -> Tuple[List[str], Dict[str, List[str]]]:
        root = os.path.join(self.root_dir, VALID_META_SPLITS[self.meta_split])
        classes = []
        class_paths = {}
        for alphabet in os.listdir(root):
            alphabet = os.path.join(root, alphabet)
            if not os.path.isdir(alphabet):
                continue
            for character in os.listdir(alphabet):
                character = os.path.join(alphabet, character)
                if not os.path.isdir(character):
                    continue
                for example in os.listdir(character):
                    if not example.endswith(".png") or example.startswith("._"):
                        continue
                    example = os.path.join(character, example)
                    try:
                        class_paths[str(character)].append(str(example))
                    except KeyError:
                        class_paths[str(character)] = [str(example)]
                classes.append(str(character))
        return classes, class_paths

    def sample_meta_batch(
        self,
        batch_size: int = 1,
        n_classes_per_task: Optional[int] = None,
        k_shots_per_class: Optional[int] = None,
        test_shots_per_class: Optional[int] = None,
        return_n_k_along_same_axis: bool = True,
        rotate_classes: bool = True,
    ):
        """
        Returns data for a classification task in a dictionary
        mapping string to tuple of omniglot {"{train, test}": (images, labels)}.
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

        # sample k_shots_per_class for each class
        meta_batch_images = []
        meta_batch_labels = []
        for i in range(batch_size):
            # sample n_classes_per_task classes:
            classes = np.random.choice(
                self.all_classes, n_classes_per_task, replace=False
            )
            task_images = []
            task_labels = []
            for j, class_name in enumerate(classes):
                if rotate_classes:
                    times_to_rotate_90 = np.random.randint(0, 4)
                paths_for_class = np.random.choice(
                    self.class_paths[class_name],
                    k_shots_per_class + test_shots_per_class,
                    replace=False,
                )
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
                    if rotate_classes:
                        img = np.rot90(img, k=times_to_rotate_90, axes=[1, 2])
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
                    1,
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
                    1,
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

        # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # train_images = train_images.to(dev)
        # train_labels = train_labels.to(dev)
        # test_images = test_images.to(dev)
        # test_labels = test_labels.to(dev)
        batch = {
            "train": (train_images, train_labels),
            "test": (test_images, test_labels),
        }
        return batch
