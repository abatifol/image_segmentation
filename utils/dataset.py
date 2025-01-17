import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from operator import add
import gc
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score
import albumentations as A
from torchsummary import summary
import tifffile
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


def preprocess_image(data_folder):
    image = tifffile.imread(data_folder)

    ## Adding a channel dimension for grayscale image
    if image.ndim == 2:
        image = image[..., np.newaxis]

    # Normalization of each pixel
    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean) / std

    # Convert to PyTorch tensor and resize
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (channel, H, W)
    transform = transforms.Compose([
        transforms.Resize((256, 256))
    ])
    image_tensor = transform(image_tensor.unsqueeze(0)).squeeze(0)

    return image_tensor

def preprocess_mask(path):
    mask = tifffile.imread(path)

    ## Adding a channel dimension for grayscale image
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]

    # Normalization of each pixel
    mask_tensor = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (channel, H, W)

    # Convert to pytoch tensor and resize
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)
    ])
    mask_tensor = transform(mask_tensor.unsqueeze(0)).squeeze(0)

    return mask_tensor
    
def augment_image(image, mask):
    
    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.permute(1, 2, 0).numpy()

    transform = A.Compose([
        A.Resize(256,256, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.RandomCrop(height=256, width=256, always_apply=True),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
    
    ])
    augmented = transform(image = image_np,mask = mask_np)
    augmented_image , augmented_mask = augmented['image'],augmented['mask']
    
    augmented_image = torch.tensor(augmented_image, dtype=torch.float32).permute(2, 0, 1)
    augmented_mask  = torch.tensor(augmented_mask,dtype=torch.float32).permute(2, 0, 1)
    
    return augmented_image,augmented_mask

## Dataset for Kidney Blood Vessel Segmentation (SenNet + Hoa kaggle competition)
class KidneyDataset(Dataset):
    def __init__(self,image_files, mask_files, input_size=(256, 256), augmentation_transforms=None):
        self.image_files=image_files
        self.mask_files=mask_files
        self.input_size=input_size
        self.augmentation_transforms=augmentation_transforms
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self,idx):
        image_path=self.image_files[idx]
        mask_path=self.mask_files[idx]
        
        image = preprocess_image(image_path)
        mask = preprocess_mask(mask_path)
        if self.augmentation_transforms:
            image,mask=self.augmentation_transforms(image,mask)
        return image,mask

##############################################################################
############              RETINA VESSEL DATASET              #################
##############################################################################

#define data augmentation to be applied on training data
transform = A.Compose([
    A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
    A.RandomCrop(height=256, width=256, always_apply=True),
    A.OneOf(
        [
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.9,
    ),
])

class RetinaDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.transform = transform

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image / 255.0
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.from_numpy(image)

        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples
