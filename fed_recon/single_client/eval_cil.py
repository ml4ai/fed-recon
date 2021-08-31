"""
Functions to evaluate a model over the single client, omniglot and mini-ImageNet continual learning benchmarks.

This code builds on the experiments proposed in [1] and goes further to compute continual learning metrics for
mini-ImageNet as the mean accuracy over all classes as they are added incrementally.

References:
    [1] Javed, K. and White, M., 2019. Meta-learning representations for continual learning.
        In Advances in Neural Information Processing Systems (pp. 1820-1830).
        https://proceedings.neurips.cc/paper/2019/hash/f4dd765c12f2ef67f98f3558c282a9cd-Abstract.html
"""
from typing import Union

from fed_recon.data.mini_imagenet.eval_cil import eval_mini_imagenet_cl
from fed_recon.data.omniglot.eval_cil import eval_omniglot_cil
from fed_recon.models.embedding_distribution_network.model import DistributionNetworkCL
from fed_recon.models.protonet.model import PrototypicalNetworkCL

SUPPORTED_DATASETS = {"mini-imagenet", "omniglot"}


def eval_cl(
    model: Union[PrototypicalNetworkCL, DistributionNetworkCL],
    root_dir: str,
    n_classes_per_task: int = 1,
    runs: int = 10,
    dataset: str = "omniglot",
    device: str = "cpu",
):
    """
    Evaluate a model on the continual learning mini-ImageNet benchmark in which all 20 meta-test classes are seen in
    procession. At each new class, the model is evaluated on the accuracy of distinguishing the current class and
    all previously seen classes.

    :param model: A nn.Module with an `evaluate` method which will learn the current training class(es) and
        compute the accuracy on the union of all previously seen classes and the current set of classes
    :param root_dir: Directory containing the mini-ImageNet data.
    :param n_classes_per_task: how many classes are learned on each task.
    :param runs: How many independent runs.
    :param device: cuda or cpu
    :return: Meta-test test-set accuracy for each class increment, for each run.
    """
    if dataset == "mini-imagenet":
        return eval_mini_imagenet_cl(
            model=model,
            root_dir=root_dir,
            n_classes_per_task=n_classes_per_task,
            runs=runs,
            device=device,
        )
    elif dataset == "omniglot":
        return eval_omniglot_cil(
            model=model,
            root_dir=root_dir,
            n_classes_per_task=n_classes_per_task,
            runs=runs,
            device=device,
        )
    else:
        raise NotImplementedError(f"{dataset} not in {SUPPORTED_DATASETS}")
