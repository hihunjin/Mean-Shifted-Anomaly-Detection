import os
from typing import Callable, List, Optional, Union
from PIL import Image

import numpy as np
import torch

from .base import AnomalyDataset


def save_npz(filename: str, **kwargs):
    with open(filename, 'wb') as f:
        print("Saving to", filename)
        np.savez(f, **kwargs)


def load_npz(filename: str):
    import numpy as np

    return np.load(filename)


class Cars3D(AnomalyDataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_type: Union[List[str], str] = "attr",
    ) -> None:
        super().__init__(root=root, split=split, transform=transform, target_type=target_type)
        assert split in ["train", "test"]
        assert target_type in ["attr"]

        if split == "train":
            data = load_npz(os.path.join(root, "cars3d__az_id__train.npz"))
            self.imgs = data["imgs"]
        else:
            data = load_npz(os.path.join(root, "cars3d__az_id__test.npz"))
            self.imgs = data["imgs"]
        self.attr_names = ["contents", "classes"]  # car type, z_direction(pose?)
        self.filename = []
        self.mask_filename = []
        self.attr = []
        contents = data["contents"]
        classes = data["classes"]

        contents = np.isin(contents, range(5))
        classes = np.isin(classes, range(4))
        self.attr = np.stack([contents, classes], axis=1, dtype=np.int64)
        self.attr = torch.tensor(self.attr, dtype=torch.int64)

    def load_image(self, index: int) -> Image:
        return Image.fromarray(self.imgs[index])

    def __len__(self) -> int:
        return len(self.imgs)


def cars_3d(
    root: str = "/NFS/database_personal/anomaly_detection/data/Cars3D",
    transform: Optional[Callable] = None,
    split: str = "train",
    target_type: str = "attr",
    **kwargs,
):
    return Cars3D(
        root=root,
        split=split,
        transform=transform,
        target_type=target_type,
    )


def _cars_3d(
    root: str = "/NFS/database_personal/anomaly_detection/data/Cars3D",
    transform: Optional[Callable] = None,
    target_type: str = "attr",
):
    train_dataset = Cars3D(
        root=root,
        split="train",
        transform=transform,
        target_type=target_type,
    )
    test_dataset = Cars3D(
        root=root,
        split="test",
        transform=transform,
        target_type=target_type,
    )

    return train_dataset, test_dataset




DATASET_TYPES = [
    cars_3d,
]
