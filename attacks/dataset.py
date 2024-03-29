"""Module for dataset related functions"""
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
# from .flowers import Flowers17
from flowers import Flowers17
from functools import partial
# from .indoor67 import Indoor67
from indoor67 import Indoor67


torch.multiprocessing.set_sharing_strategy("file_system")
num_workers = 4
ds_root = "../datasets/"


def get_weights(ds, n_classes):
    weights = []
    for _, y in ds:
        if y < n_classes:
            weights.append(1)
        else:
            weights.append(0)
    return weights


ds_choices = [
    "mnist",
    "fake_28",
    "emnist_letters",
    "kmnist",
    "fashion",
    "cifar10",
    "cifar100",
    "flowers17",
    "tiny_images",
    "imagenet",
    "indoor67",
    "imagenet_tiny",
    "svhn",
    "svhn_28",
    "gtsrb",
]


nclasses_dict = {
    "mnist": 10,
    "fake_28": 10,
    "emnist_letters": 27,
    "kmnist": 10,
    "fashion": 10,
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
    "svhn_28": 10,
    "flowers17": 17,
    "imagenet": 1000,
    "imagenet_tiny": 1000,
    "tiny_images": 10,
    "gtsrb": 43,
    "indoor67": 67,
}

nch_dict = {
    "mnist": 1,
    "fake_28": 1,
    "emnist_letters": 1,
    "kmnist": 1,
    "fashion": 1,
    "cifar10": 3,
    "cifar100": 3,
    "svhn": 3,
    "svhn_28": 1,
    "flowers17": 3,
    "imagenet": 3,
    "imagenet_tiny": 3,
    "tiny_images": 3,
    "gtsrb": 3,
    "indoor67": 3,
}

xdim_dict = {
    "mnist": 28,
    "fake_28": 28,
    "emnist_letters": 28,
    "kmnist": 28,
    "fashion": 28,
    "cifar10": 32,
    "cifar100": 32,
    "svhn": 32,
    "svhn_28": 28,
    "flowers17": 224,
    "imagenet": 224,
    "indoor67": 224,
    "imagenet_tiny": 32,
    "tiny_images": 32,
    "gtsrb": 32,
}

ds_dict = {
    "mnist": datasets.MNIST,
    "fake_28": partial(datasets.FakeData, image_size=(1, 28, 28), num_classes=10),
    "kmnist": datasets.KMNIST,
    "fashion": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "flowers17": Flowers17,
    # "imagenet": ImageNet1k,
    "svhn": datasets.SVHN,
    "svhn_28": datasets.SVHN,
    "indoor67": Indoor67,
}


transforms_augment = {
    "mnist": [transforms.RandomResizedCrop(28)],
    "fake_28": [transforms.RandomResizedCrop(28)],
    "emnist_letters": [transforms.RandomResizedCrop(28)],
    "kmnist": [transforms.RandomResizedCrop(28)],
    "fashion": [transforms.RandomResizedCrop(28)],
    "cifar10": [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ],
    "cifar100": [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ],
    "svhn": [transforms.RandomCrop(32, padding=4)],
    "svhn_28": [transforms.RandomCrop(28, padding=4)],
    "flowers17": [
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
    ],
    "imagenet": [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()],
    "indoor67": [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()],
    "imagenet_tiny": [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
    ],
    "tiny_images": [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ],
    "gtsrb": [transforms.Resize((32, 32)), transforms.RandomCrop(32, padding=4),],
}

transforms_normalize = {
    "mnist": [transforms.Normalize(mean=[0.5], std=[0.5])],
    "fake_28": [transforms.Normalize(mean=[0.5], std=[0.5])],
    "emnist_letters": [transforms.Normalize(mean=[0.5], std=[0.5])],
    "kmnist": [transforms.Normalize(mean=[0.5], std=[0.5])],
    "fashion": [transforms.Normalize(mean=[0.5], std=[0.5])],
    "cifar10": [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
    "cifar100": [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
    "svhn": [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
    "svhn_28": [transforms.Normalize(mean=[0.5], std=[0.5])],
    "flowers17": [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
    "imagenet": [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
    "indoor67": [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
    "imagenet_tiny": [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
    "tiny_images": [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
    "gtsrb": [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
}

transforms_standardize = {
    "mnist": [transforms.Normalize(mean=[0.1307], std=[0.3081])],
    "kmnist": [transforms.Normalize(mean=[0.1918], std=[0.3385])],
    "fashion": [transforms.Normalize(mean=[0.2860], std=[0.3205])],
    "cifar10": [
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
    ],
    "cifar100": [
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        )
    ],
    "svhn": [
        transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]
        )
    ],
    "flowers17": [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ],
    "imagenet": [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ],
    "tiny_images": [
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
    ],
}


def get_dataloaders(
    ds: str = "cifar10",
    batch_size: int = 256,
    augment: bool = False,
    standardize: bool = False,
    shuffle=True,
    n_classes=None,
) -> Tuple[DataLoader, DataLoader]:
    """returns train and test loaders"""

    transform_train_list, transform_test_list = [], []
    if ds in ["tiny_images"]:
        transform_train_list = [transforms.ToPILImage()]
        transform_test_list = [transforms.ToPILImage()]
    elif ds in ["indoor67", "imagenet", "flowers17"]:
        transform_train_list = [transforms.RandomResizedCrop(224)]
        transform_test_list = [transforms.RandomResizedCrop(224)]
    elif ds in ["svhn_28"]:
        transform_train_list = [transforms.Resize(28), transforms.Grayscale()]
        transform_test_list = [transforms.Resize(28), transforms.Grayscale()]

    if augment:
        transform_train_list += transforms_augment[ds]

    transform_train_list += [transforms.ToTensor()]
    transform_test_list += [transforms.ToTensor()]

    if standardize:
        transform_train_list += transforms_standardize[ds]
        transform_test_list += transforms_standardize[ds]
    else:
        transform_train_list += transforms_normalize[ds]
        transform_test_list += transforms_normalize[ds]

    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose(transform_test_list)

    if ds in ["svhn", "svhn_28"]:
        dataset_train = ds_dict[ds](
            root=f"{ds_root}/svhn",
            split="train",
            transform=transform_train,
            download=True,
        )
        dataset_test = ds_dict[ds](
            root=f"{ds_root}/svhn", split="test", transform=transform_test,
        )
    elif ds in ["fake_28"]:
        dataset_train = ds_dict[ds](size=50000, transform=transform_train)
        dataset_test = ds_dict[ds](size=10000, transform=transform_test)
    else:
        if ds in ["imagenet_tiny"]:
            ds = "imagenet"
        dataset_train = ds_dict[ds](
            root=f"{ds_root}/{ds}",
            train=True,
            transform=transform_train,
            download=True,
        )
        dataset_test = ds_dict[ds](
            root=f"{ds_root}/{ds}", train=False, transform=transform_test,
        )

    shuffle_train = shuffle
    shuffle_test = True
    if ds in ["tiny_images", "imagenet", "imagenet_tiny", "indoor67"]:
        shuffle_train = False

    if n_classes is None:
        sampler_train = sampler_test = None
    else:
        shuffle_train = shuffle_test = False
        weights_train = get_weights(dataset_train, n_classes)
        weights_test = get_weights(dataset_test, n_classes)

        sampler_train = torch.utils.data.sampler.WeightedRandomSampler(
            weights_train, len(weights_train)
        )
        sampler_test = torch.utils.data.sampler.WeightedRandomSampler(
            weights_test, len(weights_test)
        )

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=False,
        sampler=sampler_train,
    )
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler_test,
    )

    return dataloader_train, dataloader_test

