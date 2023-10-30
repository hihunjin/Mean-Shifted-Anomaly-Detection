import os
from typing import Callable, List, Optional, Union

import torch

from .base import AnomalyDataset

PRODUCT_LIST = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class MVTecAD(AnomalyDataset):
    def __init__(
        self,
        root: str,
        split: str,
        product: str,
        transform: Optional[Callable] = None,
        target_type: Union[List[str], str] = "attr",
        **kwargs,
    ) -> None:
        super().__init__(root=root, split=split, transform=transform, target_type=target_type)
        assert split in ["train", "test"]
        assert target_type in ["attr", "mask"]

        self.attr_names = os.listdir(os.path.join(root, product, "ground_truth"))
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


def mvtec_ad(
    root: str = "/NFS/database_personal/anomaly_detection/data/MVTec_AD",
    split: str = "train",
    product: str = "bottle",
    transform: Optional[Callable] = None,
    target_type: str = "attr",
):
    return MVTecAD(root=root, split=split, product=product, transform=transform, target_type=target_type)


def mvtec_ad_bottle(**kwargs):
    return mvtec_ad(product="bottle", **kwargs)


def mvtec_ad_cable(**kwargs):
    return mvtec_ad(product="cable", **kwargs)


def mvtec_ad_capsule(**kwargs):
    return mvtec_ad(product="capsule", **kwargs)


def mvtec_ad_carpet(**kwargs):
    return mvtec_ad(product="carpet", **kwargs)


def mvtec_ad_grid(**kwargs):
    return mvtec_ad(product="grid", **kwargs)


def mvtec_ad_hazelnut(**kwargs):
    return mvtec_ad(product="hazelnut", **kwargs)


def mvtec_ad_leather(**kwargs):
    return mvtec_ad(product="leather", **kwargs)


def mvtec_ad_metal_nut(**kwargs):
    return mvtec_ad(product="metal_nut", **kwargs)


def mvtec_ad_pill(**kwargs):
    return mvtec_ad(product="pill", **kwargs)


def mvtec_ad_screw(**kwargs):
    return mvtec_ad(product="screw", **kwargs)


def mvtec_ad_tile(**kwargs):
    return mvtec_ad(product="tile", **kwargs)


def mvtec_ad_toothbrush(**kwargs):
    return mvtec_ad(product="toothbrush", **kwargs)


def mvtec_ad_transistor(**kwargs):
    return mvtec_ad(product="transistor", **kwargs)


def mvtec_ad_wood(**kwargs):
    return mvtec_ad(product="wood", **kwargs)


def mvtec_ad_zipper(**kwargs):
    return mvtec_ad(product="zipper", **kwargs)


DATASET_TYPES = [
    mvtec_ad,
    mvtec_ad_bottle,
    mvtec_ad_cable,
    mvtec_ad_capsule,
    mvtec_ad_carpet,
    mvtec_ad_grid,
    mvtec_ad_hazelnut,
    mvtec_ad_leather,
    mvtec_ad_metal_nut,
    mvtec_ad_pill,
    mvtec_ad_screw,
    mvtec_ad_tile,
    mvtec_ad_toothbrush,
    mvtec_ad_transistor,
    mvtec_ad_wood,
    mvtec_ad_zipper,
]
