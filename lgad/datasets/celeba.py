from typing import Callable, Optional

from torchvision.datasets import CelebA


def celeba(
    root: str = "/NFS/database_personal/anomaly_detection/data/CelebA",
    split: str = "train",
    transform: Optional[Callable] = None,
    target_type: str = "attr",
    **kwargs,
):
    return CelebA(root=root, split=split, transform=transform, target_type=target_type)


DATASET_TYPES = [
    celeba,
]
