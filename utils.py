from typing import Callable, List, Optional, Union, Dict

import clip
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import models as M
import torchvision.models as models
import torch.nn.functional as F
from PIL import ImageFilter
import random
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_resnet18 = transforms.Compose([
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


moco_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


from models.base import image_transform

class Transform:
    def __init__(self, n_px: int = None, original: bool = True):
        if n_px is None:
            n_px = 224
        if original:
            resize_crop = [transforms.RandomResizedCrop(224, scale=(0.2, 1.))]
        else:
            resize_crop = image_transform(n_px).transforms[:-1]
        self.moco_transform = transforms.Compose([
            *resize_crop,
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2


class Model(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if backbone == 152:
            self.backbone_type = "resnet"
            self.backbone = models.resnet152(pretrained=True)
        elif backbone == "clip":
            self.backbone_type = "clip"
            self.backbone = clip.load("ViT-B/32", jit=False)[0]
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.fc = torch.nn.Linear(512, 512)
        else:
            self.backbone_type = "resnet"
            self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, backbone, train_fc=False)

    def forward(self, x):
        if self.backbone_type == "resnet":
            z1 = self.backbone(x)
            z_n = F.normalize(z1, dim=-1)
            return z_n
        else:
            with torch.no_grad():
                image_features = self.backbone.encode_image(x)
            return self.fc(image_features.to(torch.float32))

def freeze_parameters(model, backbone, train_fc=False):
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
    if backbone == 152:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False
        for p in model.layer2.parameters():
            p.requires_grad = False



def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def make_condition(attr_names: List[str], attribute_config: Dict[Union[int, str], bool]) -> Callable:
    attr_names_lower = [name.lower() for name in attr_names]
    new_attribute_config = {
        k if isinstance(k, int) else attr_names_lower.index(k.lower()): v
        for k, v in attribute_config.items()
    }
    # str, int가 중복되어서 들어오는 것을 방지
    assert len(attribute_config.keys()) == len(new_attribute_config.keys()), "Duplicate keys"

    return lambda labels: all(labels[i] if v else not labels[i] for i, v in new_attribute_config.items())


def make_subset(dataset, cond: Callable):
    from torch.utils.data import Subset

    return Subset(dataset, [i for i, labels in enumerate(dataset.attr) if cond(labels)])


def get_condition_config(dataset_name: str, dataset_attr_names=None):
    if dataset_name == "two_class_color_mnist":
        TRAIN_CONDITION_CONFIG = TEST_CONDITION_CONFIG = {
            "01234": True,
            "red": True,
        }
    elif dataset_name == "multi_color_mnist":
        TRAIN_CONDITION_CONFIG = TEST_CONDITION_CONFIG = {
            "label": True,
            "color": True,
        }
    elif dataset_name == "waterbirds":
        TRAIN_CONDITION_CONFIG = TEST_CONDITION_CONFIG = dict(zip(["y", "place"], [False, False]))
    elif dataset_name == "celeba":
        assert dataset_attr_names is not None
        _temp = {
            dataset_attr_names[15]: True,  # Eyeglasses
            dataset_attr_names[39]: True,  # Young

            dataset_attr_names[22]: True,  # Mustache
            # dataset_attr_names[31]: True,  # Smiling
        }
        TRAIN_CONDITION_CONFIG = TEST_CONDITION_CONFIG = _temp


    return TRAIN_CONDITION_CONFIG, TEST_CONDITION_CONFIG


def get_loaders(dataset, label_class, batch_size, backbone):
    if dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10
        transform = transform_color if backbone == 152 else transform_resnet18
        coarse = {}
        trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
        testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        trainset_1 = ds(root='data', train=True, download=True, transform=Transform(), **coarse)
        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        trainset_1.data = trainset_1.data[idx]
        trainset_1.targets = [trainset_1.targets[i] for i, flag in enumerate(idx, 0) if flag]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                                  drop_last=False)
        return train_loader, test_loader, torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                                                      shuffle=True, num_workers=2, drop_last=False)
    elif dataset in [
        "two_class_color_mnist",
        "multi_color_mnist",
        "waterbirds",
        "celeba",
    ]:
        from datasets.builder import build_dataset

        model, preprocess = M.load("CLIP/ViT-B/16")

        trainset = build_dataset(dataset, "train", preprocess)
        testset = build_dataset(dataset, "test", preprocess)
        trainset_1 = build_dataset(_target_=dataset, split="train", transform=Transform(model.visual.input_resolution))

        train_condition_config = get_condition_config(
            dataset_name=dataset,
            dataset_attr_names=trainset.attr_names,
        )[0]
        train_condition = make_condition(trainset_1.attr_names, train_condition_config)

        train_subset = make_subset(trainset, train_condition)
        train_1_subset = make_subset(trainset_1, train_condition)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                                  drop_last=False)
        return train_loader, test_loader, torch.utils.data.DataLoader(train_1_subset, batch_size=batch_size,
                                                                      shuffle=True, num_workers=2, drop_last=False)

    else:
        print('Unsupported Dataset')
        exit()