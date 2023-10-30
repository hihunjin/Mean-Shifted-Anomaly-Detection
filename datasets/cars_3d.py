import os
from typing import Callable, List, Optional, Union

import numpy as np
import torch

from .base import AnomalyDataset


def save_npz(filename: str, **kwargs):
    with open(filename, 'wb') as f:
        print("Saving to", filename)
        np.savez(f, **kwargs)
        np.savez_compressed(f, **kwargs)


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
        else:
            data = load_npz(os.path.join(root, "cars3d__az_id__test.npz"))
        self.attr_names = ["classes", "anom_label"]
        self.filename = []
        self.mask_filename = []
        self.attr = []
        for attr_name in os.listdir(os.path.join(root, product, split)):
            for filename in os.listdir(os.path.join(root, product, split, attr_name)):
                self.filename.append(os.path.join(product, split, attr_name, filename))
                attr = [0] * len(self.attr_names)
                if attr_name != "good":
                    attr[self.attr_names.index(attr_name)] = 1
                self.attr.append(attr)
        self.attr = torch.tensor(self.attr, dtype=torch.int64)


def cars_3d(
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
    
]
