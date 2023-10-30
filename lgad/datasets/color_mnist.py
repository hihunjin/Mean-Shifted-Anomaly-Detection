from typing import Callable, List, Optional, Union, Dict

import numpy as np

import torch
from torch.utils.data import Subset
import torchvision

from PIL import Image

from .base import AnomalyDataset


def _train_valid_split(dataset, num_train, num_valid, seed=42):
    rng = torch.Generator().manual_seed(seed)
    train_idxs, valid_idxs = [], []

    for i in range(10):
        class_idxs = (dataset.targets == i).nonzero(as_tuple=False).flatten().tolist()
        perm_class_idxs = torch.randperm(len(class_idxs), generator=rng).tolist()

        train_idxs.extend([class_idxs[idx] for idx in perm_class_idxs[:num_train]])
        valid_idxs.extend([class_idxs[idx] for idx in perm_class_idxs[num_train:num_train+num_valid]])

    train_dataset = Subset(dataset, sorted(train_idxs))
    valid_dataset = Subset(dataset, sorted(valid_idxs))
    return train_dataset, valid_dataset


class ColorMNIST(AnomalyDataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_type: Union[List[str], str] = "attr",
        only_two_classes: bool = False,
        attribute_label_only: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            target_type=target_type,
            attribute_label_only=attribute_label_only,
        )
        assert split in ["train", "valid", "test"]
        assert target_type in ["attr"]

        self.attr_names: List[str] = ["01234", "red"]
        self.dataset = []
        self.attr = []

        if split == "train":
            num_of_data = [4000 for _ in range(10)]
        elif split == "valid":
            num_of_data = [1000 for _ in range(10)]
        else:
            num_of_data = [800 for _ in range(10)]

        if only_two_classes:
            num_of_data = [num_of_data[idx] if idx in [0, 5] else 0 for idx in range(10)]

        dataset = torchvision.datasets.MNIST(
            root=root,
            train=False if split == "test" else True,
            download=True,
        )

        if split in ["train", "valid"]:
            train_dataset, valid_dataset = _train_valid_split(dataset, 4000, 1000, seed=42)
            dataset = train_dataset if split == "train" else valid_dataset

        cnt = [0 for _ in range(10)]
        for img, label in dataset:
            if cnt[label] >= num_of_data[label]:
                continue

            attr = [0] * len(self.attr_names)
            attr[0] = True if label < 5 else False

            img = np.pad(np.array(img), ((28, 28), (28, 28)), "constant", constant_values=0)
            img = np.stack([img] * 3, axis=-1)
            if cnt[label] >= (num_of_data[label] // 2):
                img[:, :, [1, 2]] = 0
                attr[1] = True
            else:
                img[:, :, [0, 2]] = 0
                attr[1] = False
            img = Image.fromarray(img)
            self.dataset.append(img)
            self.attr.append(attr)
            cnt[label] += 1

        self.attr = torch.tensor(self.attr, dtype=torch.int64)

    def load_image(self, index: int) -> Image.Image:
        image = self.dataset[index]
        return image

    def __len__(self) -> int:
        return len(self.dataset)


def coloring_inplace(img: np.ndarray, color: str) -> np.ndarray:
    if color == "red":
        img[:, :, [1, 2]] = 0
    elif color == "green":
        img[:, :, [0, 2]] = 0
    elif color == "blue":
        img[:, :, [0, 1]] = 0
    else:
        raise ValueError(f"Color {color} is not supported.")
    return img


class MultiColorMNIST(AnomalyDataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_type: Union[List[str], str] = "attr",
        multi: Optional[Dict] = None,
        attribute_label_only: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            target_type=target_type,
            attribute_label_only=attribute_label_only,
        )
        assert split in ["train", "valid", "test"]
        assert target_type in ["attr"]
        if multi is None:
            multi = {
                "attribute_config": {
                    "label": {
                        0: True,
                        1: True,
                        2: True,
                        3: True,
                        4: True,
                        5: False,
                        6: False,
                        7: False,
                        8: False,
                        9: False,
                    },
                    "color": {
                        "red": True,
                        "green": False,
                        "blue": False,
                    },
                },
            }

        self.attr_names: List[str] = list(multi["attribute_config"].keys())
        self.dataset = []
        self.attr = []

        if split == "train":
            num_of_data = [4500 for _ in range(10)]
        elif split == "valid":
            num_of_data = [900 for _ in range(10)]
        else:
            num_of_data = [870 for _ in range(10)]

        for i in range(10):
            if i not in multi["attribute_config"]["label"] or multi["attribute_config"]["label"][i] is None:
                num_of_data[i] = 0

        dataset = torchvision.datasets.MNIST(
            root=root,
            train=False if split == "test" else True,
            download=True,
        )

        if split in ["train", "valid"]:
            train_dataset, valid_dataset = _train_valid_split(dataset, 4500, 900, seed=42)
            dataset = train_dataset if split == "train" else valid_dataset

        cnt = [0 for _ in range(10)]
        for img, label_digit in dataset:
            if cnt[label_digit] >= num_of_data[label_digit]:
                continue

            attr = [0] * len(self.attr_names)
            attr[0] = multi["attribute_config"]["label"][label_digit]  # True if label_digit < 5 else False

            img = np.pad(np.array(img), ((28, 28), (28, 28)), "constant", constant_values=0)
            img = np.stack([img] * 3, axis=-1)
            color = list(multi["attribute_config"]["color"].keys())[
                cnt[label_digit] % len(multi["attribute_config"]["color"])
            ]
            coloring_inplace(img, color)
            attr[1] = multi["attribute_config"]["color"][color]

            img = Image.fromarray(img)
            self.dataset.append(img)
            self.attr.append(attr)
            cnt[label_digit] += 1

        self.attr = torch.tensor(self.attr, dtype=torch.int64)

    def load_image(self, index: int) -> Image.Image:
        image = self.dataset[index]
        return image

    def __len__(self) -> int:
        return len(self.dataset)


# def color_mnist(
#     root: str = "/NFS/database_personal/anomaly_detection/data/MNIST",
#     split: str = "train",
#     transform: Optional[Callable] = None,
#     target_type: str = "attr",
#     attribute_label_only: bool = False,
#     **kwargs,
# ):
#     return ColorMNIST(
#         root=root,
#         split=split,
#         transform=transform,
#         target_type=target_type,
#         attribute_label_only=attribute_label_only,
#     )


def two_class_color_mnist(
    root: str = "/NFS/database_personal/anomaly_detection/data/MNIST",
    split: str = "train",
    transform: Optional[Callable] = None,
    target_type: str = "attr",
    attribute_label_only: bool = False,
    **kwargs,
):
    return ColorMNIST(
        root=root,
        split=split,
        transform=transform,
        target_type=target_type,
        only_two_classes=True,
        attribute_label_only=attribute_label_only,
    )


def multi_color_mnist(
    root: str = "/NFS/database_personal/anomaly_detection/data/MNIST",
    split: str = "train",
    transform: Optional[Callable] = None,
    target_type: str = "attr",
    multi: Optional[Dict] = None,
    attribute_label_only: bool = False,
):
    return MultiColorMNIST(
        root=root,
        split=split,
        transform=transform,
        target_type=target_type,
        multi=multi,
        attribute_label_only=attribute_label_only,
    )


def color_mnist(*args, **kwargs):
    return multi_color_mnist(*args, **kwargs)


DATASET_TYPES = [
    color_mnist,
    two_class_color_mnist,
    multi_color_mnist,
]
