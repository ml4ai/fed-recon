import operator
from typing import Tuple


class EarlyStopper:
    """
    Computes stopping criterion given a metric and a patience.
    """

    def __init__(
        self,
        patience: int = 20,
        metric_should_increase: bool = True,
        min_steps: int = 0,
    ):
        """
        Args:
            patience: How many evaluations to continue training if eval does not improve.
            metric_should_increase: If True, metric is expected to increase (i.e. set to True for accuracy or IoU,
                False for a loss function such as cross entropy.
        """
        self.patience = patience
        self.metric_should_increase = metric_should_increase
        if metric_should_increase:
            self.eval_operator = operator.gt
        else:
            self.eval_operator = operator.lt
        self._best_metric = None
        self._best_num_steps = None
        self.num_evals_without_improving = 0
        self.min_steps = min_steps
        if min_steps > 0:
            self._best_num_steps = min_steps
        print("Built EarlyStopper with patience {}".format(self.patience))

    def continue_training(
        self, metric: float, total_steps_taken: int
    ) -> Tuple[bool, bool]:
        """Returns two bools indicating whether we should continue training and whether this is the best eval seen yet."""
        if total_steps_taken <= self.min_steps:
            self._best_metric = metric
            return True, True
        elif self._best_metric is None or self.eval_operator(metric, self._best_metric):
            self.num_evals_without_improving = 0
            self._best_metric = metric
            self._best_num_steps = total_steps_taken
            return True, True
        else:
            self.num_evals_without_improving += 1
            if self.num_evals_without_improving > self.patience:
                return False, False
        return True, False

    def best_metric(self):
        return self._best_metric

    def best_num_steps(self):
        return self._best_num_steps
