"""
Functions to evaluate a model over the mini-ImageNet continual learning benchmark.

This code builds on the experiments proposed in [1] and goes further to compute continual learning metrics for
mini-ImageNet as the mean accuracy over all classes as they are added incrementally.

References:
    [1] Javed, K. and White, M., 2019. Meta-learning representations for continual learning.
        In Advances in Neural Information Processing Systems (pp. 1820-1830).
        https://proceedings.neurips.cc/paper/2019/hash/f4dd765c12f2ef67f98f3558c282a9cd-Abstract.html
"""
import copy
from typing import Tuple

import numpy as np

from mtm.data.mini_imagenet.mini_imagenet import MiniImagenet
from mtm.models.protonet.model import PrototypicalNetworkCL


def eval_mini_imagenet_cl(
    model: PrototypicalNetworkCL,
    root_dir: str,
    n_classes_per_task: int = 1,
    k: int = 30,
    test_shots_per_class: int = 30,
    runs: int = 100,
    image_resize_hw: Tuple[int, int] = (84, 84),
    device: str = "cpu",
):
    """
    Eval class incremental learning on mini-ImageNet
    :param model: A nn.Module with an `evaluate` method which will learn the current training class(es) and
        compute the accuracy on the union of all previously seen classes and the current set of classes
    :param root_dir: Directory containing the mini-ImageNet data.
    :param n_classes_per_task:
    :param k:
    :param test_shots_per_class: Number of test/query points to evaluate on.
        Following [1], we sample an equal number of query examples as training examples for each new class by default.
    :param runs:
    :param image_resize_hw:
    :return:
    """
    n_meta_test_classes = 20
    dataset = MiniImagenet(
        root_dir=root_dir,
        n_classes_per_task=n_meta_test_classes,
        k_shots_per_class=k,
        test_shots_per_class=test_shots_per_class,
        meta_split="test",
    )
    assert dataset.meta_splits == {"test"}

    all_accs = []
    all_n_classes = []
    for i in range(runs):
        all_data = dataset.sample_meta_batch(
            1,
            meta_split="test",
            n_classes_per_task=n_meta_test_classes,
            k_shots_per_class=k,
            test_shots_per_class=k,
            return_n_k_along_same_axis=False,
            sample_k_value=False,
        )  # -> ([b, n, k, channels, rows, cols], [b, n, k])
        model_i = copy.deepcopy(model)
        print(f"Evaluating {k}-shot class-incremental learning run {i} of {runs}")
        for j in range(0, n_meta_test_classes, n_classes_per_task):
            train_images, train_labels = all_data["train"]
            train_images = train_images[:, j : j + n_classes_per_task, :, :, :, :]
            train_labels = train_labels[:, j : j + n_classes_per_task]
            train_images = train_images.reshape(
                [1, n_classes_per_task * k, 3, *image_resize_hw]
            )
            train_labels = train_labels.reshape([1, n_classes_per_task * k])

            test_images, test_labels = all_data["test"]
            test_images = test_images[:, 0 : j + n_classes_per_task, :, :, :, :]
            test_labels = test_labels[:, 0 : j + n_classes_per_task]
            test_images = test_images.reshape(
                [1, (j + 1) * n_classes_per_task * k, 3, *image_resize_hw]
            )
            test_labels = test_labels.reshape([1, (j + 1) * n_classes_per_task * k])

            train_images.to(device=device)
            train_labels.to(device=device)
            test_images.to(device=device)
            test_labels.to(device=device)

            acc = model_i.evaluate(
                train_inputs=train_images,
                train_labels=train_labels,
                test_inputs=test_images,
                test_labels=test_labels,
                n_train_classes=n_classes_per_task,
                k_shots=k,
            )
            print(
                f"Class-incremental test-set accuracy over {j + 1} meta-test tasks: {np.round(acc.item(), 2)}"
            )
            all_accs.append(acc.item())
            all_n_classes.append(j + 1)

    print(
        f"Final continual learning, class-incremental accuracy averaged across tasks: {np.mean(all_accs)}"
    )

    return all_accs, all_n_classes
