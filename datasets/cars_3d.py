# import os
# from typing import Callable, List, Optional, Union

# import torch

# from .base import AnomalyDataset


# class Cars3D(AnomalyDataset):
#     def __init__(
#         self,
#         root: str,
#         split: str,
#         transform: Optional[Callable] = None,
#         target_type: Union[List[str], str] = "attr",
#     ) -> None:
#         super().__init__(root=root, split=split, transform=transform, target_type=target_type)
#         assert split in ["train", "test"]
#         assert target_type in ["attr"]

#         self.attr_names = os.listdir(os.path.join(root, product, split, "ground_truth"))
#         self.filename = []
#         self.mask_filename = []
#         self.attr = []
#         for attr_name in os.listdir(os.path.join(root, product, split)):
#             for filename in os.listdir(os.path.join(root, product, split, attr_name)):
#                 self.filename.append(os.path.join(product, split, attr_name, filename))
#                 attr = [0] * len(self.attr_names)
#                 if attr_name != "good":
#                     attr[self.attr_names.index(attr_name)] = 1
#                 self.attr.append(attr)
#         self.attr = torch.tensor(self.attr, dtype=torch.int64)


# def cars_3d(
#     root: str = "/NFS/database_personal/anomaly_detection/data/Cars3D",
#     transform: Optional[Callable] = None,
#     target_type: str = "attr",
# ):
#     train_dataset = Cars3D(
#         root=root,
#         split="train",
#         transform=transform,
#         target_type=target_type,
#     )
#     test_dataset = Cars3D(
#         root=root,
#         split="test",
#         transform=transform,
#         target_type=target_type,
#     )

#     return train_dataset, test_dataset




# DATASET_TYPES = [
    
# ]
