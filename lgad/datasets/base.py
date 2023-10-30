import os
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Subset
from torchvision.datasets import CelebA
from torchvision.datasets.vision import VisionDataset

from PIL import Image


__all__ = [
    "AnomalyDataset",
    "make_condition",
    "make_subset",
    "convert_anomaly_label",
    "sample_index",
]


class AnomalyDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_type: Union[List[str], str] = "attr",
        attribute_label_only: bool = False,
    ) -> None:
        super().__init__(root=root, transform=transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        self.filename: List[str] = []
        self.mask_filename: List[str] = []
        self.attr: torch.Tensor = None
        self.attr_names: List[str] = []
        self.attribute_label_only: bool = attribute_label_only

    def load_image(self, index: int) -> Image.Image:
        return Image.open(os.path.join(self.root, self.filename[index]))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if not self.attribute_label_only:
            X = self.load_image(index)

        target: List[Any] = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "mask":
                raise NotImplementedError
            else:
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None and not self.attribute_label_only:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]
        else:
            target = None

        if self.attribute_label_only:
            return torch.Tensor([]), target
        else:
            return X, target

    def __len__(self) -> int:
        return len(self.filename)


def make_condition(attr_names: List[str], attribute_config: Dict[Union[int, str], bool]) -> Callable:
    attr_names_lower = [name.lower() for name in attr_names]
    new_attribute_config = {
        k if isinstance(k, int) else attr_names_lower.index(k.lower()): v
        for k, v in attribute_config.items()
    }
    # str, int가 중복되어서 들어오는 것을 방지
    assert len(attribute_config.keys()) == len(new_attribute_config.keys()), "Duplicate keys"

    return lambda labels: all(labels[i] if v else not labels[i] for i, v in new_attribute_config.items())


def make_subset(dataset: Union[AnomalyDataset, CelebA], cond: Callable):
    return Subset(dataset, [i for i, labels in enumerate(dataset.attr) if cond(labels)])


def convert_anomaly_label(labels: torch.Tensor, cond: Callable):
    return torch.tensor([1 if cond(label) else 0 for label in labels])


def sample_index(
    attrs: Tensor,  # [N] or [N, num_attrs]
    n_samples: int,
    per_attr: bool = True,  # if False, sample from all attributes (if n_samples is int)
    seed: int = 0,
    sort: bool = True,
) -> Tensor:
    rng = torch.Generator().manual_seed(seed)

    if per_attr:
        if attrs.ndim == 1:
            attrs = attrs.unsqueeze(dim=1)

        num_attrs = attrs.size(1)
        values_per_attr = [attrs[:, i].unique().tolist() for i in range(num_attrs)]
        indicies_list = []

        for values in product(*values_per_attr):
            mask = torch.ones_like(attrs[:, 0], dtype=torch.bool)
            for i, v in enumerate(values):
                mask &= attrs[:, i] == v

            candidates = mask.nonzero(as_tuple=False).flatten()
            _indices = torch.randperm(candidates.size(0), generator=rng)[:n_samples]
            indicies_list.append(candidates[_indices])

        indices = torch.cat(indicies_list)

    else:
        indices = torch.randperm(attrs.size(0), generator=rng)[:n_samples]

    indices = indices.to(attrs.device)
    if sort:
        indices = indices.sort().values

    return indices
