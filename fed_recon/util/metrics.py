from typing import Union, List

import numpy as np
import torch


def get_accuracy(
    predictions: torch.FloatTensor, labels: torch.LongTensor
) -> torch.FloatTensor:
    """
    Compute the accuracy of predictions against labels.
    :param predictions: torch.Tensor of model predictions of shape [batch_size, n_examples]
    :param labels: torch.Tensor of labels of shape [batch_size, n_examples]
    :return: `torch.FloatTensor` instance of mean accuracy
    """
    return torch.mean(predictions.eq(labels).float())


def ci95(a: Union[List[float], np.ndarray]):
    """Computes the 95% confidence interval of the array `a`."""
    sigma = np.std(a)
    return 1.96 * sigma / np.sqrt(len(a))
