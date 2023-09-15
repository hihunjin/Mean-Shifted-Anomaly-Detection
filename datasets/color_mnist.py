from typing import Callable, List, Optional, Union, Dict

import numpy as np
import torch
import torchvision
from PIL import Image

from .base import AnomalyDataset


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
        assert split in ["train", "test"]
        assert target_type in ["attr"]

        self.attr_names: List[str] = ["01234", "red"]
        self.dataset = []
        self.attr = []

        num_of_data = [2000 if split == "train" else 800 for _ in range(10)]
        if only_two_classes:
            num_of_data = [num_of_data[idx] if idx in [0, 5] else 0 for idx in range(10)]

        mnist_dataset = torchvision.datasets.MNIST(
            root=root,
            train=True if split == "train" else False,
            download=True,
        )
        cnt = [0 for _ in range(10)]
        for img, label in mnist_dataset:
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
        assert split in ["train", "test"]
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

        num_of_data = [2000 if split == "train" else 800 for _ in range(10)]
        for i in range(10):
            if i not in multi["attribute_config"]["label"] or multi["attribute_config"]["label"][i] is None:
                num_of_data[i] = 0

        mnist_dataset = torchvision.datasets.MNIST(
            root=root,
            train=True if split == "train" else False,
            download=True,
        )

        cnt = [0 for _ in range(10)]
        for img, label_digit in mnist_dataset:
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


def color_mnist(
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
        attribute_label_only=attribute_label_only,
    )


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


DATASET_TYPES = [
    color_mnist,
    two_class_color_mnist,
    multi_color_mnist,
]
