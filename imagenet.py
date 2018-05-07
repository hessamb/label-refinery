"""Dataset class for loading imagenet data."""

import os

from torch.utils import data as data_utils
from torchvision import datasets as torch_datasets
from torchvision import transforms


def get_train_loader(imagenet_path, batch_size, num_workers):
    train_dataset = ImageNet(imagenet_path, is_train=True)
    return data_utils.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
        num_workers=num_workers)


def get_val_loader(imagenet_path, batch_size, num_workers):
    val_dataset = ImageNet(imagenet_path, is_train=False)
    return data_utils.DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True,
        num_workers=num_workers)


class ImageNet(torch_datasets.ImageFolder):
    """Dataset class for ImageNet dataset.

    Arguments:
        root_dir (str): Path to the dataset root directory, which must contain
            train/ and val/ directories.
        is_train (bool): Whether to read training or validation images.
    """
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, root_dir, is_train):
        if is_train:
            root_dir = os.path.join(root_dir, 'train')
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(ImageNet.MEAN, ImageNet.STD),
            ])
        else:
            root_dir = os.path.join(root_dir, 'val')
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(ImageNet.MEAN, ImageNet.STD),
            ])
        super().__init__(root_dir, transform=transform)
