import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import transforms

torch.manual_seed(1)


def get_a_train_transform():
    """Get transformer for training data

    Returns:
        Compose: Composed transformations
    """
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.09, scale_limit=0.09, rotate_limit=7, p=0.5),
        # A.Affine(scale=0.9, translate_percent=(-0.1, 0.1), rotate=7, shear=[-7, 7], cval=0, fit_output=False, p=0.2),
        # A.RandomResizedCrop(height=28, width=28, scale=(0.8, 1.0), p=0.5),
        A.RandomBrightnessContrast(p=0.5),
                # A.GaussNoise(p=0.2),
                # A.Equalize(p=0.2),
        A.Normalize(mean=(0.1307,), std=(0.3081,)),
        ToTensorV2(),
    ])


def get_a_test_transform():
    """Get transformer for test data

    Returns:
        Compose: Composed transformations
    """
    return A.Compose([
        A.Normalize(mean=(0.1307,), std=(0.3081,)),
        ToTensorV2(),
    ])


def get_p_train_transform():
    """Get Pytorch Transform function for train data

    Returns:
        Compose: Composed transformations
    """
    random_rotation_degree = 5
    img_size = (28, 28)
    random_crop_percent = (0.85, 1.0)
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, random_crop_percent),
        transforms.RandomRotation(random_rotation_degree),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_p_test_transform():
    """Get Pytorch Transform function for test data

    Returns:
        Compose: Composed transformations
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
