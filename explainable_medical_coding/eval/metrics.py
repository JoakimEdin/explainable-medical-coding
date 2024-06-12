from abc import abstractmethod
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
import pandas as pd

from explainable_medical_coding.utils.tensor import detach
from explainable_medical_coding.utils.tokenizer import TargetTokenizer


class Metric:
    higher_is_better = True
    batch_update = True
    filter_codes = True
    metric_type = ""

    def __init__(
        self,
        name: str,
        number_of_classes: int,
    ):
        self.name = name
        self.number_of_classes = number_of_classes
        self.device = "cpu"
        self.reset()

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any):
        """Update the metric from a batch"""
        raise NotImplementedError()

    def set_target_boolean_indices(self, target_boolean_indices: list[bool]):
        self.target_boolean_indices = target_boolean_indices

    @abstractmethod
    def compute(
        self,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ):
        """Compute the metric value"""
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """Reset the metric"""
        raise NotImplementedError()

    def to(self, device: str):
        self.device = device
        self.reset()
        return self

    def copy(self):
        return deepcopy(self)

    def set_number_of_classes(self, number_of_classes: int):
        self.number_of_classes = number_of_classes


class MetricCollection:
    def __init__(
        self,
        metrics: list[Metric],
        code_indices: Optional[torch.Tensor] = None,
        code_system_name: Optional[str] = None,
        autoregressive: bool = False,
        threshold: Optional[float] = 0.5,
        sos_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = 1,
        pad_token_id: Optional[int] = 2,
    ):
        self.metrics = metrics
        self.code_system_name = code_system_name
        self.code_system_name = code_system_name
        self.autoregressive = autoregressive
        self.threshold = threshold
        if code_indices is not None:
            # Get overlapping indices
            self.code_indices = code_indices.clone()
            self.set_number_of_classes(len(code_indices))
        else:
            self.code_indices = None
        self.reset()

    def set_number_of_classes(self, number_of_classes_split: int):
        """Sets the number of classes for metrics with the filter_codes attribute to the number of classes in the split.
        Args:
            number_of_classes_split (int): Number of classes in the split
        """
        for metric in self.metrics:
            if metric.filter_codes:
                metric.set_number_of_classes(number_of_classes_split)

    def to(self, device: str):
        self.metrics = [metric.to(device) for metric in self.metrics]
        if self.code_indices is not None:
            self.code_indices = self.code_indices.to(device)
        return self

    def filter_tensor(
        self, tensor: torch.Tensor, code_indices: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if code_indices is None:
            return tensor
        return torch.index_select(tensor, -1, code_indices)

    def is_best(
        self,
        prev_best: Optional[torch.Tensor],
        current: torch.Tensor,
        higher_is_better: bool,
    ) -> bool:
        if higher_is_better:
            return prev_best is None or current > prev_best
        else:
            return prev_best is None or current < prev_best

    def update_best_metrics(self, metric_dict: dict[str, torch.Tensor]):
        for metric in self.metrics:
            if metric.name not in metric_dict:
                continue

            if self.is_best(
                self.best_metrics[metric.name],
                metric_dict[metric.name],
                metric.higher_is_better,
            ):
                self.best_metrics[metric.name] = metric_dict[metric.name]

    def get_prediction(self, y_probs: torch.Tensor) -> torch.Tensor:
        if self.autoregressive:
            preds = y_probs.argmax(dim=-1)
            return torch.zeros_like(y_probs[:, 0, :]).scatter_(-1, preds, 1.0)
        return (y_probs > self.threshold).long()

    def one_hot_targets(
        self, y_probs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.autoregressive:
            # one hot encode targets
            targets = torch.zeros_like(y_probs[:, 0, :]).scatter_(-1, targets, 1.0)
        return targets

    def update_classification_metrics(
        self, y_probs: torch.Tensor, targets: torch.Tensor
    ):
        predictions = self.get_prediction(y_probs)
        targets = self.one_hot_targets(y_probs, targets)

        # filter codes
        predictions = self.filter_tensor(predictions, self.code_indices)
        targets = self.filter_tensor(targets, self.code_indices)

        for metric in self.metrics:
            if metric.metric_type == "classification" and metric.batch_update:
                metric.update(predictions, targets)

    def update_ranking_metrics(self, y_probs: torch.Tensor, targets: torch.Tensor):
        targets = self.one_hot_targets(y_probs, targets)

        # filter codes
        predictions = self.filter_tensor(y_probs, self.code_indices)
        targets = self.filter_tensor(targets, self.code_indices)

        for metric in self.metrics:
            if metric.metric_type == "ranking" and metric.batch_update:
                metric.update(predictions, targets)

    def update_loss(self, loss: torch.Tensor):
        for metric in self.metrics:
            if metric.metric_type == "loss":
                metric.update(loss)

    def update(
        self,
        y_probs: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
    ):
        y_probs, targets = (
            detach(y_probs),
            detach(targets),
        )
        if loss is not None:
            loss = detach(loss).cpu()
            self.update_loss(loss)

        self.update_classification_metrics(y_probs, targets)
        self.update_ranking_metrics(y_probs, targets)

    def compute(
        self,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        metric_dict = {
            metric.name: metric.compute()
            for metric in self.metrics
            if metric.batch_update
        }
        if y_probs is not None and targets is not None:
            # Compute the metrics for the whole dataset
            if self.code_indices is not None:
                y_probs_filtered = self.filter_tensor(y_probs, self.code_indices.cpu())
                targets_filtered = self.filter_tensor(targets, self.code_indices.cpu())

            for metric in self.metrics:
                if metric.batch_update:
                    continue
                if metric.filter_codes and self.code_indices is not None:
                    metric_dict[metric.name] = metric.compute(
                        y_probs=y_probs_filtered, targets=targets_filtered
                    )
                else:
                    metric_dict[metric.name] = metric.compute(
                        y_probs=y_probs, targets=targets
                    )

            metric_dict.update(
                {
                    metric.name: metric.compute(y_probs=y_probs, targets=targets)
                    for metric in self.metrics
                    if not metric.batch_update
                }
            )
        self.update_best_metrics(metric_dict)
        return metric_dict

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    def reset(self):
        self.reset_metrics()
        self.best_metrics = {metric.name: None for metric in self.metrics}

    def get_best_metric(self, metric_name: str) -> dict[str, torch.Tensor]:
        return self.best_metrics[metric_name]

    def copy(self):
        return deepcopy(self)

    def set_threshold(self, threshold: float):
        self.threshold = threshold


""" ------------Classification Metrics-------------"""


class ExactMatchRatio(Metric):
    metric_type = "classification"

    def __init__(
        self,
        name: str = "exact_match_ratio",
        number_of_classes: int = 0,
        filter_codes: bool = True,
    ):
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(
            name=name,
            number_of_classes=number_of_classes,
        )
        self.filter_codes = filter_codes

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        self._num_exact_matches += torch.all(
            torch.eq(predictions, targets), dim=-1
        ).sum()
        self._num_examples += targets.size(0)

    def compute(
        self,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._num_exact_matches / self._num_examples

    def reset(self):
        self._num_exact_matches = torch.tensor(0).to(self.device)
        self._num_examples = 0


class Recall(Metric):
    metric_type = "classification"

    def __init__(
        self,
        number_of_classes: int,
        average: str = "micro",
        name: str = "recall",
        filter_codes: bool = True,
    ):
        if average:
            name = f"{name}_{average}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(
            name=name,
            number_of_classes=number_of_classes,
        )
        self._average = average
        self.filter_codes = filter_codes

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        self._tp += torch.sum(predictions * targets, dim=0)
        self._fn += torch.sum((1 - predictions) * targets, dim=0)

    def compute(
        self,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._average == "micro":
            return (self._tp.sum() / (self._tp.sum() + self._fn.sum() + 1e-10)).cpu()
        if self._average == "macro":
            return torch.mean(self._tp / (self._tp + self._fn + 1e-10)).cpu()
        if self._average is None or self._average == "none":
            return (self._tp / (self._tp + self._fn + 1e-10)).cpu()
        raise ValueError(f"Invalid average: {self._average}")

    def reset(self):
        self._tp = torch.zeros((self.number_of_classes)).to(self.device)
        self._fn = torch.zeros((self.number_of_classes)).to(self.device)


class Precision(Metric):
    metric_type = "classification"

    def __init__(
        self,
        number_of_classes: int,
        average: str = "micro",
        name: str = "precision",
        filter_codes: bool = True,
    ):
        if average:
            name = f"{name}_{average}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(
            name=name,
            number_of_classes=number_of_classes,
        )
        self._average = average
        self.filter_codes = filter_codes

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        self._tp += torch.sum(predictions * targets, dim=0)
        self._fp += torch.sum((predictions) * (1 - targets), dim=0)

    def compute(
        self,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ):
        if self._average == "micro":
            return (self._tp.sum() / (self._tp.sum() + self._fp.sum() + 1e-10)).cpu()
        if self._average == "macro":
            return torch.mean(self._tp / (self._tp + self._fp + 1e-10)).cpu()
        if self._average is None or self._average == "none":
            return (self._tp / (self._tp + self._fp + 1e-10)).cpu()
        raise ValueError(f"Invalid average: {self._average}")

    def reset(self):
        self._tp = torch.zeros((self.number_of_classes)).to(self.device)
        self._fp = torch.zeros((self.number_of_classes)).to(self.device)


class FPR(Metric):
    higher_is_better = False
    metric_type = "classification"

    def __init__(
        self,
        number_of_classes: int,
        average: str = "micro",
        name: str = "fpr",
        filter_codes: bool = True,
    ):
        if average:
            name = f"{name}_{average}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(
            name=name,
            number_of_classes=number_of_classes,
        )
        self._average = average
        self.filter_codes = filter_codes

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        self._fp += torch.sum(predictions * (1 - targets), dim=0)
        self._tn += torch.sum((1 - predictions) * (1 - targets), dim=0)

    def compute(
        self,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._average == "micro":
            return (self._fp.sum() / (self._fp.sum() + self._tn.sum() + 1e-10)).cpu()
        if self._average == "macro":
            return torch.mean(self._fp / (self._fp + self._tn + 1e-10)).cpu()
        if self._average is None or self._average == "none":
            return (self._fp / (self._fp + self._tn + 1e-10)).cpu()
        raise ValueError(f"Invalid average: {self._average}")

    def reset(self):
        self._fp = torch.zeros((self.number_of_classes)).to(self.device)
        self._tn = torch.zeros((self.number_of_classes)).to(self.device)


class AUC(Metric):
    batch_update = False
    metric_type = "classification"

    def __init__(
        self,
        number_of_classes: int,
        average: str = "micro",
        name: str = "auc",
        filter_codes: bool = True,
    ):
        """Area under the ROC curve. All classes that have no positive examples are ignored as implemented by Mullenbach et al. Please note that all the y_probs and targets are stored in the GPU memory if they have not already been moved to the CPU.

        Args:
            number_of_classes (int): number of classes.
            average (str, optional): type of averaging to perform. Can be "micro", "macro", "none". Defaults to "micro".
            name (str, optional): name of the metric. Defaults to "auc".
            filter_codes (bool, optional): whether to filter out codes that have no positive examples. Defaults to True.
        """
        if average:
            name = f"{name}_{average}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(name=name, number_of_classes=number_of_classes)
        self._average = average
        self.filter_codes = filter_codes

    def compute(
        self,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> np.float32:
        if y_probs is None or targets is None:
            raise ValueError("y_probs and targets must be provided to calculate AUC.")
        y_probs = detach(y_probs).numpy()
        targets = detach(targets).numpy()
        if self._average == "micro":
            fprs, tprs, _ = self.roc_curve(
                y_probs=y_probs, targets=targets, average=self._average
            )
            value = auc(fprs, tprs)
        if self._average == "macro":
            fprs, tprs, _ = self.roc_curve(
                y_probs=y_probs, targets=targets, average="none"
            )
            value = np.mean([auc(fpr, tpr) for fpr, tpr in zip(fprs, tprs)])
        return value

    def update(self, y_probs: torch.Tensor, targets: torch.Tensor):
        raise NotImplementedError("AUC is not batch updateable.")

    def roc_curve(
        self,
        y_probs: torch.Tensor,
        targets: torch.Tensor,
        average: str = "micro",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        thresholds = torch.linspace(0, 1, 1000)

        if average == "micro":
            return roc_curve(targets.ravel(), y_probs.ravel())
        if average == "none":
            fprs, tprs, thresholds = [], [], []
            for i in range(targets.shape[1]):
                if targets[:, i].sum() == 0:
                    continue

                fpr, tpr, threshold = roc_curve(targets[:, i], y_probs[:, i])

                if np.any(np.isnan(fpr)) or np.any(np.isnan(tpr)):
                    continue

                fprs.append(fpr)
                tprs.append(tpr)
                thresholds.append(threshold)

            return fprs, tprs, thresholds
        raise ValueError(f"Invalid average: {average}")

    def reset(self):
        pass


class F1Score(Metric):
    metric_type = "classification"

    def __init__(
        self,
        number_of_classes: int,
        average: str = "micro",
        name: str = "f1",
        filter_codes: bool = True,
    ):
        if average:
            name = f"{name}_{average}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(
            name=name,
            number_of_classes=number_of_classes,
        )
        self._average = average
        self.filter_codes = filter_codes

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        self._tp += torch.sum((predictions) * (targets), dim=0)
        self._fp += torch.sum(predictions * (1 - targets), dim=0)
        self._fn += torch.sum((1 - predictions) * targets, dim=0)

    def compute(
        self,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ):
        if self._average == "micro":
            return (
                self._tp.sum()
                / (self._tp.sum() + 0.5 * (self._fp.sum() + self._fn.sum()) + 1e-10)
            ).cpu()
        if self._average == "macro":
            return torch.mean(
                self._tp / (self._tp + 0.5 * (self._fp + self._fn) + 1e-10)
            ).cpu()
        if self._average is None or self._average == "none":
            return (self._tp / (self._tp + 0.5 * (self._fp + self._fn) + 1e-10)).cpu()
        raise ValueError(f"Invalid average: {self._average}")

    def reset(self):
        self._tp = torch.zeros((self.number_of_classes)).to(self.device)
        self._fp = torch.zeros((self.number_of_classes)).to(self.device)
        self._fn = torch.zeros((self.number_of_classes)).to(self.device)


""" ------------Information Retrieval Metrics-------------"""


class PrecisionAtRecall(Metric):
    metric_type = "ranking"

    def __init__(
        self,
        number_of_classes: int,
        name: str = "precision@recall",
        filter_codes: bool = True,
    ):
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(name=name, number_of_classes=number_of_classes)
        self.filter_codes = filter_codes

    def update(self, y_probs: torch.Tensor, targets: torch.Tensor):
        num_targets = targets.sum(dim=1, dtype=torch.int64)
        _, indices = torch.sort(y_probs, dim=1, descending=True)
        sorted_targets = targets.gather(1, indices)
        sorted_targets_cum = torch.cumsum(sorted_targets, dim=1)
        self._precision_sum += torch.sum(
            sorted_targets_cum.gather(1, num_targets.unsqueeze(1) - 1).squeeze()
            / num_targets
        )
        self._num_examples += y_probs.size(0)

    def compute(
        self,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._precision_sum.cpu() / self._num_examples

    def reset(self):
        self._num_examples = 0
        self._precision_sum = torch.tensor(0.0).to(self.device)


class Precision_K(Metric):
    metric_type = "ranking"

    def __init__(
        self,
        number_of_classes: int,
        k: int = 10,
        name: str = "precision",
        filter_codes: bool = True,
    ):
        name = f"{name}@{k}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(name=name, number_of_classes=number_of_classes)
        self._k = k
        self.filter_codes = filter_codes

    def update(self, y_probs: torch.Tensor, targets: torch.Tensor):
        top_k = torch.topk(y_probs, dim=1, k=self._k)

        targets_k = targets.gather(1, top_k.indices)
        y_probs_k = torch.ones(targets_k.shape, device=targets_k.device)

        tp = torch.sum(y_probs_k * targets_k, dim=1)
        fp = torch.sum((y_probs_k) * (1 - targets_k), dim=1)
        self._num_examples += y_probs.size(0)
        self._precision_sum += torch.sum(tp / (tp + fp + 1e-10))

    def compute(
        self,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._precision_sum.cpu() / self._num_examples

    def reset(self):
        self._num_examples = 0
        self._precision_sum = torch.tensor(0.0).to(self.device)


class MeanAveragePrecision(Metric):
    metric_type = "ranking"

    def __init__(
        self,
        number_of_classes: int,
        name: str = "map",
        filter_codes: bool = True,
    ):
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(name=name, number_of_classes=number_of_classes)
        self.filter_codes = filter_codes

    def update(self, y_probs: torch.Tensor, targets: torch.Tensor):
        _, indices = torch.sort(y_probs, dim=1, descending=True)
        sorted_targets = targets.gather(1, indices)
        sorted_targets_cum = torch.cumsum(sorted_targets, dim=1)
        batch_size = y_probs.size(0)
        denom = torch.arange(1, targets.shape[1] + 1, device=targets.device).repeat(
            batch_size, 1
        )
        prec_at_k = sorted_targets_cum / denom
        average_precision_batch = torch.sum(
            prec_at_k * sorted_targets, dim=1
        ) / torch.sum(sorted_targets, dim=1)
        self._average_precision_sum += torch.sum(average_precision_batch)
        self._num_examples += batch_size

    def compute(
        self,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._average_precision_sum.cpu() / self._num_examples

    def reset(self):
        self._num_examples = 0
        self._average_precision_sum = torch.tensor(0.0).to(self.device)


class Recall_K(Metric):
    metric_type = "ranking"

    def __init__(
        self,
        number_of_classes: int,
        k: int = 10,
        name: str = "recall",
        filter_codes: bool = True,
    ):
        name = f"{name}@{k}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(name=name, number_of_classes=number_of_classes)
        self._k = k
        self.filter_codes = filter_codes

    def update(self, y_probs: torch.Tensor, targets: torch.Tensor):
        top_k = torch.topk(y_probs, dim=1, k=self._k)

        targets_k = targets.gather(1, top_k.indices)
        y_probs_k = torch.ones(targets_k.shape, device=targets_k.device)

        tp = torch.sum(y_probs_k * targets_k, dim=1)
        total_number_of_relevant_targets = torch.sum(targets, dim=1)

        self._num_examples += y_probs.size(0)
        self._recall_sum += torch.sum(tp / (total_number_of_relevant_targets + 1e-10))

    def compute(
        self,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._recall_sum.cpu() / self._num_examples

    def reset(self):
        self._num_examples = 0
        self._recall_sum = torch.tensor(0.0).to(self.device)


""" ------------Running Mean Metrics-------------"""


class RunningMeanMetric(Metric):
    metric_type = "running_mean"

    def __init__(
        self,
        name: str,
        number_of_classes: int,
    ):
        """Create a running mean metric.

        Args:
            name (str): Name of the metric
            number_of_classes (Optional[int], optional): Number of classes. Defaults to None.
        """
        super().__init__(name=name, number_of_classes=number_of_classes)

    def update(self, batch: dict):
        raise NotImplementedError

    def update_value(
        self,
        values: torch.Tensor,
        reduce_by: torch.Tensor,
        weight_by: torch.Tensor,
    ):
        """
        Args:
            values (torch.Tensor): Values of the metric
            reduce_by (Optional[torch.Tensor], optional): A single or per example divisor of the values. Defaults to batch size.
            weight_by (Optional[torch.Tensor], optional): A single or per example weights for the running mean. Defaults to `reduce_by`.
        """
        values = detach(values)
        reduce_by = detach(reduce_by)

        numel = values.numel() if isinstance(values, torch.Tensor) else 1
        value = values.sum().tolist()

        reduce_by = (
            reduce_by.sum().tolist()
            if isinstance(reduce_by, torch.Tensor)
            else (reduce_by or numel)
        )

        weight_by = (
            weight_by.sum().tolist()
            if isinstance(weight_by, torch.Tensor)
            else (weight_by or reduce_by)
        )

        values = value / reduce_by

        d = self.weight_by + weight_by
        w1 = self.weight_by / d
        w2 = weight_by / d

        self._values: torch.Tensor = (
            self._values * w1 + values * w2
        )  # Reduce between batches (over entire epoch)

        self.weight_by: torch.Tensor = d

    def compute(
        self,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._values

    def reset(self):
        self._values: torch.Tensor = torch.tensor(0.0).to(self.device)
        self.weight_by: torch.Tensor = torch.tensor(0.0).to(self.device)


class LossMetric(RunningMeanMetric):
    higher_is_better = False
    metric_type = "loss"

    def __init__(
        self,
        number_of_classes: int,
        name: str = "loss",
        filter_codes: bool = False,
    ):
        super().__init__(
            name=name,
            number_of_classes=number_of_classes,
        )
        self.filter_codes = filter_codes

    def update(self, loss: torch.Tensor):
        self.update_value(loss, reduce_by=loss.numel(), weight_by=loss.numel())


def eval_per_code(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_tokenizer: TargetTokenizer,
    code2description: dict[str, str],
) -> pd.DataFrame:
    """Calculates the precision, recall, and f1 score for each code in the dataset. Returns the results as a pandas DataFrame.

    Args:
        predictions (torch.Tensor): model predictions (batch_size, num_classes)
        targets (torch.Tensor): target labels (batch_size, num_classes)
        target_tokenizer (TargetTokenizer): target tokenizer
        code2description (dict[str, str]): dictionary mapping code to description

    Returns:
        pd.DataFrame: dataframe containing the precision, recall, and f1 score for each code in the dataset
    """
    tp = (predictions * targets).sum(axis=0)
    fp = (predictions * (1 - targets)).sum(axis=0)
    fn = ((1 - predictions) * targets).sum(axis=0)

    target_counts = targets.sum(axis=0)
    targets_nonzero = target_counts > 0

    tp = tp[targets_nonzero]
    fp = fp[targets_nonzero]
    fn = fn[targets_nonzero]

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    code_names = np.array(target_tokenizer.id2target)[targets_nonzero.numpy()]
    code_description = [code2description[code] for code in code_names]

    return pd.DataFrame(
        {
            "code": code_names,
            "code_description": code_description,
            "precision": precision.numpy(),
            "recall": recall.numpy(),
            "f1": f1.numpy(),
            "tp": tp.numpy(),
            "fp": fp.numpy(),
            "fn": fn.numpy(),
            "target_count": target_counts.numpy()[targets_nonzero.numpy()],
        }
    )
