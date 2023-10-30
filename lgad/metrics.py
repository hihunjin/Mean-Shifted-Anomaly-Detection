from typing import Union, Literal, Callable, TypeVar, Sequence, Tuple, overload
from typing_extensions import Concatenate, ParamSpec, TypedDict

import numpy as np
from sklearn.metrics import precision_recall_curve

from torch import Tensor
from torcheval.metrics import functional as MF


__all__ = [
    "BinaryMetrics",
    "mean_std",
    "metric_mean_std",
    "align_device",
    "optimal_threshold",
    "binary_f1_score",
    "binary_accuracy",
    "binary_auroc",
    "binary_auprc",
    "binary_fpr95",
    "binary_metrics",
]


P = ParamSpec("P")
R = TypeVar("R")


class BinaryMetrics(TypedDict, total=False):
    auroc: float
    auprc: float
    accuracy: float
    f1: float
    fpr95: float


def mean_std(
    values: Sequence[float],
) -> Tuple[float, float]:
    return np.mean(values).item(), np.std(values).item()


def metric_mean_std(
    metrics: Sequence[BinaryMetrics],
) -> Tuple[BinaryMetrics, BinaryMetrics]:
    mean = BinaryMetrics()
    std = BinaryMetrics()

    for k in metrics[0].keys():
        values = [metric[k] for metric in metrics]
        mean[k] = np.mean(values).item()
        std[k] = np.std(values).item()
    return mean, std


def align_device(
    func: Callable[Concatenate[Tensor, Tensor, P], R],
) -> Callable[Concatenate[Tensor, Tensor, P], R]:
    def wrapper(input: Tensor, target: Tensor, *args: P.args, **kwargs: P.kwargs) -> R:
        if input.device != target.device:
            input = input.to(target.device)
        return func(input, target, *args, **kwargs)
    return wrapper


def optimal_threshold(
    input: Tensor,
    target: Tensor,
) -> float:
    y_true, y_pred = target.cpu(), input.cpu()
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1 = 2 * precision * recall / np.maximum(precision + recall, np.finfo(float).eps)
    threshold = thresholds[np.argmax(f1)].item()
    return threshold


@align_device
def binary_f1_score(
    input: Tensor,
    target: Tensor,
    *,
    threshold: Union[float, Literal["auto"]] = "auto",
) -> float:
    if threshold == "auto":
        threshold = optimal_threshold(input, target)
    return MF.binary_f1_score(input, target, threshold=threshold).item()


@align_device
def binary_accuracy(
    input: Tensor,
    target: Tensor,
    *,
    threshold: Union[float, Literal["auto"]] = "auto",
) -> float:
    if threshold == "auto":
        threshold = optimal_threshold(input, target)
    return MF.binary_accuracy(input, target, threshold=threshold).item()


@align_device
def binary_auroc(
    input: Tensor,
    target: Tensor,
) -> float:
    return MF.binary_auroc(input, target).item()


@align_device
def binary_auprc(
    input: Tensor,
    target: Tensor,
) -> float:
    return MF.binary_auprc(input, target).item()


# https://github.com/deeplearning-wisc/MCM/blob/main/utils/detection_util.py
def _stable_cumsum(arr, rtol: float = 1e-05, atol: float = 1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError("cumsum was found to be unstable: its last element does not correspond to sum")
    return out


# https://github.com/deeplearning-wisc/MCM/blob/main/utils/detection_util.py
def _fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None) -> float:
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.flipud(np.argsort(y_score, kind="mergesort"))
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = _stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return (fps[cutoff] / (np.sum(np.logical_not(y_true)))).item()   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def binary_fpr95(
    input: Tensor,
    target: Tensor,
):
    y_score, y_true = input.cpu().float().numpy(), target.cpu().float().numpy()
    return _fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=1)


def _binary_metrics(
    input: Tensor,
    target: Tensor,
    *,
    threshold: Union[float, Literal["auto"]] = "auto",
    types: Sequence[str] = ("auroc", "auprc", "accuracy", "f1", "fpr95"),
) -> BinaryMetrics:
    input = input.float()

    if threshold == "auto" and ("f1" in types or "accuracy" in types):
        threshold = optimal_threshold(input, target)

    metrics = BinaryMetrics()
    if "auroc" in types:
        metrics["auroc"] = binary_auroc(input, target)
    if "auprc" in types:
        metrics["auprc"] = binary_auprc(input, target)
    if "accuracy" in types:
        metrics["accuracy"] = binary_accuracy(input, target, threshold=threshold)
    if "f1" in types:
        metrics["f1"] = binary_f1_score(input, target, threshold=threshold)
    if "fpr95" in types:
        metrics["fpr95"] = binary_fpr95(input, target)
    return metrics


@overload
def binary_metrics(
    input: Tensor,
    target: Tensor,
    *,
    threshold: Union[float, Literal["auto"]] = "auto",
    types: Sequence[str] = ("auroc", "auprc", "accuracy", "f1", "fpr95"),
) -> BinaryMetrics:
    ...


@overload
def binary_metrics(
    input: Sequence[Tensor],
    target: Tensor,
    *,
    threshold: Union[float, Literal["auto"]] = "auto",
    types: Sequence[str] = ("auroc", "auprc", "accuracy", "f1", "fpr95"),
) -> Tuple[BinaryMetrics, BinaryMetrics]:
    ...


def binary_metrics(
    input: Union[Tensor, Sequence[Tensor]],
    target: Tensor,
    *,
    threshold: Union[float, Literal["auto"]] = "auto",
    types: Sequence[str] = ("auroc", "auprc", "accuracy", "f1", "fpr95"),
) -> Union[BinaryMetrics, Tuple[BinaryMetrics, BinaryMetrics]]:
    if isinstance(input, Tensor):
        metrics = _binary_metrics(input, target, threshold=threshold, types=types)
        return metrics
    else:
        mean, std = metric_mean_std([
            _binary_metrics(input_, target, threshold=threshold, types=types)
            for input_ in input
        ])
        return mean, std
