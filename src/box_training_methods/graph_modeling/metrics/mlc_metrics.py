from typing import Iterable, Optional, List, Tuple, Union, Dict, Any

import torch
from sklearn.metrics import average_precision_score

__all__ = ["Metric",
           "Average",
           "MeanAvgPrecision",
           "MicroAvgPrecision"]


class Metric(object):
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """

    supports_distributed = False

    def __call__(
        self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions.
        gold_labels : `torch.Tensor`, required.
            A tensor corresponding to some gold label to evaluate against.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        raise NotImplementedError

    def get_metric(self, reset: bool):
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)



class Average(Metric):
    """
    This [`Metric`](./metric.md) breaks with the typical `Metric` API and just stores values that were
    computed in some fashion outside of a `Metric`.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    `Metric` API.
    """

    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    def __call__(self, value):
        """
        # Parameters

        value : `float`
            The value to average.
        """
        self._count += 1
        self._total_value += float(list(self.detach_tensors(value))[0])

    def get_metric(self, reset: bool = False):
        """
        # Returns

        The average of all values that were passed to `__call__`.
        """

        average_value = self._total_value / self._count if self._count > 0 else 0.0
        if reset:
            self.reset()
        return float(average_value)

    def reset(self):
        self._total_value = 0.0
        self._count = 0


class MeanAvgPrecision(Average):

    """Docstring for MeanAvgPrecision. """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor) -> None:  # type: ignore
        # we are goint to modify scores
        # if it is already on cpu we can to create a copy first

        if not predictions.is_cuda:
            predictions = predictions.clone()
        labels, scores = [
            t.cpu().numpy()
            for t in self.detach_tensors(gold_labels, predictions)
        ]

        for single_example_labels, single_example_scores in zip(
            labels, scores
        ):
            avg_precision = average_precision_score(
                single_example_labels, single_example_scores
            )
            super().__call__(avg_precision)


class MicroAvgPrecision(Metric):

    """Docstring for MicroAvgPrecision."""

    def __init__(self) -> None:
        super().__init__()
        self.predicted: List[torch.Tensor] = []
        self.gold: List[torch.Tensor] = []

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor) -> None:  # type: ignore
        # predictions, gold_labels: (batch_size, labels)
        predictions, gold_labels = [
            t.detach().clone().cpu()
            for t in self.detach_tensors(predictions, gold_labels)
        ]
        self.predicted.append(predictions)
        self.gold.append(gold_labels)

    def get_metric(self, reset: bool) -> float:
        micro_precision_score = -1

        if reset:
            predicted = torch.cat(self.predicted, dim=0)
            gold = torch.cat(self.gold, dim=0)
            labels, scores = [
                t.cpu().numpy() for t in self.detach_tensors(gold, predicted)
            ]
            micro_precision_score = average_precision_score(
                labels, scores, average="micro"
            )

            self.reset()

        return float(micro_precision_score)

    def reset(self) -> None:
        self.predicted = []
        self.gold = []



if __name__ == "__main__":
    micro_map = MicroAvgPrecision()
    instance_map = MeanAvgPrecision()
    steps_in_epoch = 100 
    for i in range(steps_in_epoch):
        print(i)
        t = torch.rand(10, 40) # (batch, num_labels)
        labels = t > 0.5
        micro_map(t, labels)
        MAP(t, labels)
        # Don't reset during the epoch
        map_value = instance_map.get_metric(reset=False)
        # avoid computing micro_map during the epoch. It is expensive
    # Reset the metrics at the end of the epoch to get rid of the running average
    instance_map_value = instance_map.get_metric(reset=True)
    micro_map_value = micro_map.get_metric(reset=True)
