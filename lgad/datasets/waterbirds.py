import os
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .base import AnomalyDataset


class Waterbirds(AnomalyDataset):
    SPLIT_DICT = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }
    CSV_FILENAME = "metadata.csv"

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_type: Union[List[str], str] = "attr",
        attribute_label_only: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            target_type=target_type,
            attribute_label_only=attribute_label_only,
        )
        assert split in self.SPLIT_DICT.keys()
        assert target_type in ["attr"]

        metadata = pd.read_csv(os.path.join(root, self.CSV_FILENAME))

        self.attr_names = ["y", "place"]
        split = metadata["split"] == self.SPLIT_DICT[split]
        self.filename = list(metadata["img_filename"][split])
        self.attr = torch.tensor(np.array(metadata[self.attr_names][split]), dtype=torch.int64)


def waterbirds(
    root: str = "/NFS/database_personal/anomaly_detection/data/waterbird",
    split: str = "train",
    transform: Optional[Callable] = None,
    target_type: str = "attr",
    attribute_label_only: bool = False,
    **kwargs,
):
    if split == "valid":
        split = "validation"

    return Waterbirds(
        root=root,
        split=split,
        transform=transform,
        target_type=target_type,
        attribute_label_only=attribute_label_only,
    )


DATASET_TYPES = [
    waterbirds,
]
