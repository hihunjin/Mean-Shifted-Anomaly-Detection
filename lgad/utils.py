import os
import math
from pathlib import Path
from typing import Callable, Dict, Literal, Mapping, Sequence, Union, Sequence, Optional
from itertools import chain

import torch

import numpy as np


__all__ = [
    "set_gpu",
    "reload_module",
    "image_grid",
    "get_cached_features",
    "add_options",
    "build_table",
    "save_table",
]


def set_gpu(*gpu_idx: int):
    """Sets the visible GPUs for PyTorch.

    Args:
        *gpu_idx: The indices of the GPUs to use.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(idx) for idx in gpu_idx])


def reload_module(module):
    """Reloads a module and all its submodules."""
    import lgad
    import importlib
    importlib.reload(module)
    importlib.reload(lgad)


def image_grid(
    images: torch.Tensor,
    title: str = "",
    figsize: tuple = (10, 10),
):
    from matplotlib import pyplot as plt

    num_images = images.shape[0]
    num_cr = math.ceil(math.sqrt(num_images))

    fig, axes = plt.subplots(num_cr, num_cr, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if num_images > i:
            ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())  # type: ignore
        ax.axis("off")  # type: ignore

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig


def get_cached_features(
    model_name: str,
    dataset_name: str,
    condition_config: Dict,
    model_kwargs: Dict = {},
    dataset_kwargs: Dict = {},
    *,
    splits: Sequence[Literal["train", "valid", "test"]] = ("train", "valid", "test"),
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    print_fn: Callable = print,
    cache_dir: str = "./.cache",
    flush: bool = False,
):
    import json
    import hashlib

    from torch.utils.data import DataLoader

    from lgad.models.clip import load
    from lgad.datasets import build_dataset, make_condition, make_subset

    key = json.dumps(
        {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "condition_config": condition_config,
            "model_kwargs": {
                "impl": model_kwargs.get("impl"),
            },
            "dataset_kwargs": {
                "target_type": dataset_kwargs.get("target_type", "attr"),
                "multi": dataset_kwargs.get("multi", None),
                "attribute_label_only": dataset_kwargs.get("attribute_label_only", False),
            },
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    hashkey = hashlib.md5(key.encode("utf-8")).hexdigest()
    cache_path = Path(cache_dir) / hashkey

    model, preprocess = load(model_name, device=device, **model_kwargs)
    data = {}

    for split in splits:
        split_path = (cache_path / f"{split}.pt")

        if split_path.exists() and not flush:
            if verbose:
                print_fn(f"Loading {split} data from cache '{hashkey}'...")

            features, attrs = torch.load(split_path)
            data[split] = (features.to(device).float(), attrs.to(device))

        else:
            dataset = build_dataset(dataset_name, split, preprocess, **dataset_kwargs)

            if split == "train":
                condition = make_condition(dataset.attr_names, condition_config)
                subset = make_subset(dataset, condition)
            else:
                subset = dataset

            dataloader = DataLoader(subset, batch_size=64, num_workers=4, shuffle=False)

            if verbose:
                print_fn(f"{split} set size: {len(subset)} ({len(subset) / len(dataset) * 100:.2f}%)")

                from tqdm.auto import tqdm
                load_iter = tqdm(dataloader, desc=f"Computing {split} features", leave=False)
            else:
                load_iter = dataloader

            with torch.no_grad():
                features = torch.cat([model.encode_image(v) for v, _ in load_iter])
                attrs = torch.cat([v for _, v in dataloader])

            split_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save((features.cpu(), attrs.cpu()), split_path)
            data[split] = data[split] = (features.float(), attrs.to(device))

            if verbose:
                print_fn(f"Saved {split} data to cache '{hashkey}'")

    return model, data


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options


def build_table(
    metrics: Mapping[str, Mapping[str, Union[Mapping[str, float], Sequence[Mapping[str, float]]]]],
    group_headers: Optional[Sequence[str]] = None,
    label_headers: Optional[Sequence[str]] = None,
    types: Sequence[str] = ("auroc", "auprc", "accuracy", "f1", "fpr95"),
    floatfmt: str = ".3f",
):
    from tabulate import tabulate

    formatted = {k: {kk: {} for kk in v} for k, v in metrics.items()}
    max_num_group_cols = 1
    group_names = list(list(metrics.values())[0].keys())

    if label_headers is None:
        label_headers = list(metrics.keys())
    elif len(label_headers) != len(metrics):
        raise ValueError(f"Expected {len(metrics)} label headers, got {len(label_headers)}")

    for k, v in metrics.items():
        for kk, vv in v.items():
            if isinstance(vv, dict):
                for t in types:
                    if t not in vv:
                        raise ValueError(f"Missing metric '{t}' for '{k}' / '{kk}'")
                    formatted[k][kk][t] = f"{vv[t]:{floatfmt}}"
            else:
                for t in types:
                    vs = []
                    for vvv in vv:
                        if t not in vvv:
                            raise ValueError(f"Missing metric '{t}' for '{k}' / '{kk}'")
                        vs.append(vvv[t])
                    formatted[k][kk][t] = f"{np.mean(vs):{floatfmt}} ± {np.std(vs):{floatfmt}}"

            num_group_cols = len(kk.split("/"))
            if max_num_group_cols < num_group_cols:
                max_num_group_cols = num_group_cols

    if group_headers is None:
        group_headers = [""] * max_num_group_cols
    elif len(group_headers) != max_num_group_cols:
        raise ValueError(f"Expected {max_num_group_cols} group headers, got {len(group_headers)}")

    types_headers = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "accuracy": "Accuracy",
        "f1": "F1",
        "fpr95": "FPR95",
    }

    table_label_headers = (
        [""] * max_num_group_cols +
        list(chain(*[[l.capitalize()] + [""] * (len(types)-1) for l in label_headers]))
    )
    table_metric_headers = list(group_headers) + [types_headers[t] for t in types] * len(label_headers)
    table_content = [
        [f"{v0}\n{v1}" for v0, v1 in zip(table_label_headers, table_metric_headers)]
    ]

    for group_name in group_names:
        cur_row = [v for v in group_name.split("/")]
        cur_row += [""] * (max_num_group_cols - len(cur_row))

        for k, v in formatted.items():
            if group_name in v:
                cur_row.extend(v[group_name].values())
            else:
                cur_row.extend([""] * len(types))

        table_content.append(cur_row)

    table = tabulate(
        table_content,
        headers="firstrow",
        colalign=("left",) * len(table_content[0]),
        disable_numparse=True,
    )
    return table


def save_table(table: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(table)
