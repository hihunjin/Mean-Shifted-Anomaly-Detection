from typing import Callable, Optional, Dict

from . import celeba, color_mnist, mvtec_ad, waterbirds

DATASET_TYPES = (
    celeba.DATASET_TYPES + mvtec_ad.DATASET_TYPES + waterbirds.DATASET_TYPES + color_mnist.DATASET_TYPES
)
DATASET_TYPES = {dataset_type.__name__: dataset_type for dataset_type in DATASET_TYPES}


def build_dataset(
    _target_: str,
    split: str,
    transform: Optional[Callable] = None,
    target_type: str = "attr",
    multi: Optional[Dict] = None,
    attribute_label_only: bool = False,
):
    dataset_type = DATASET_TYPES[_target_]
    return dataset_type(
        split=split,
        transform=transform,
        target_type=target_type,
        multi=multi,
        attribute_label_only=attribute_label_only,
    )
