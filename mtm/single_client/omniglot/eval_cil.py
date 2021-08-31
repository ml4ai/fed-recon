"""
Functions to evaluate a model over the omniglot continual learning benchmark.

This code builds on the experiments proposed in [1] and goes further to compute continual learning metrics for
mini-ImageNet as the mean accuracy over all classes as they are added incrementally.

References:
    [1] Javed, K. and White, M., 2019. Meta-learning representations for continual learning.
        In Advances in Neural Information Processing Systems (pp. 1820-1830).
        https://proceedings.neurips.cc/paper/2019/hash/f4dd765c12f2ef67f98f3558c282a9cd-Abstract.html
"""
import copy
from typing import Tuple, Union

import numpy as np

from mtm.data.omniglot.omniglot import Omniglot, IMAGE_RESIZE_HW
from mtm.models.embedding_distribution_network.model import DistributionNetworkCL
from mtm.models.protonet.model import PrototypicalNetworkCL


def eval_omniglot_cil(
    model: Union[PrototypicalNetworkCL, DistributionNetworkCL],
    root_dir: str,
    n_classes_per_task: int = 1,
    k: int = 15,
    test_shots_per_class: int = 5,
    runs: int = 10,
    n_meta_test_classes: int = 600,
    image_resize_hw: Tuple[int, int] = IMAGE_RESIZE_HW,
    device: str = "cuda",
):
    """
        Evaluate a model on the continual learning omniglot benchmark from [1] in which all 600 meta-test classes are seen in
    procession. At each new class, the model is evaluated on the accuracy of distinguishing the current class and
    all previously seen classes.

    :param model: A nn.Module with an `evaluate` method which will learn the current training class(es) and
        compute the accuracy on the union of all previously seen classes and the current set of classes
    :param root_dir: Directory containing the omniglot data.
    :param n_classes_per_task: how many classes to add to the model at each time step
    :param k: Number of training examples per class.
    :param test_shots_per_class: how many examples to evaluate the model on per class
    :param runs: How many independent runs.
    :param n_meta_test_classes: total number of test classes to learn
    :param image_resize_hw: dims of the input images
    :param device: whether to put tensors on the cpu or cuda
    :return: Meta-test test-set accuracy for each class increment, for each run.
    """
    dataset = Omniglot(
        root_dir=root_dir,
        n_classes_per_task=n_meta_test_classes,
        k_shots_per_class=k,
        test_shots_per_class=test_shots_per_class,
        meta_split="test",
    )

    all_accs = []
    all_n_classes = []
    for i in range(runs):
        all_data = dataset.sample_meta_batch(
            batch_size=1,
            n_classes_per_task=n_meta_test_classes,
            k_shots_per_class=k,
            test_shots_per_class=test_shots_per_class,
            return_n_k_along_same_axis=False,
            rotate_classes=False,
        )  # -> ([b, n, k, channels, rows, cols], [b, n, k])
        model_i = copy.deepcopy(model)
        print(f"Evaluating {k}-shot class-incremental learning run {i} of {runs}")
        for j in range(0, n_meta_test_classes, n_classes_per_task):
            train_images, train_labels = all_data["train"]
            train_images = train_images[:, j : j + n_classes_per_task, :, :, :, :]
            train_labels = train_labels[:, j : j + n_classes_per_task]
            train_images = train_images.reshape(
                [1, n_classes_per_task * k, 1, *image_resize_hw]
            )
            train_labels = train_labels.reshape([1, n_classes_per_task * k])

            test_images, test_labels = all_data["test"]
            test_images = test_images[:, 0 : j + n_classes_per_task, :, :, :, :]
            test_labels = test_labels[:, 0 : j + n_classes_per_task]
            test_images = test_images.reshape(
                [
                    1,
                    (j + 1) * n_classes_per_task * test_shots_per_class,
                    1,
                    *image_resize_hw,
                ]
            )
            test_labels = test_labels.reshape(
                [1, (j + 1) * n_classes_per_task * test_shots_per_class]
            )

            # FIXME: cuda implementation not working
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
