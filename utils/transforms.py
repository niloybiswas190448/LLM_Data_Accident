"""
Image transformations and augmentation utilities
"""

import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image

def get_transforms(image_size: Tuple[int, int], augmentation_config: Dict[str, Any]):
    """
    Get image transformations for training, validation, and test
    
    Args:
        image_size: Target image size (height, width)
        augmentation_config: Augmentation configuration
    
    Returns:
        Tuple of (train_transforms, val_transforms, test_transforms)
    """
    # Base transforms
    base_transforms = [
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ]
    
    # Training transforms with augmentation
    train_transforms = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5) if augmentation_config.get('train', {}).get('horizontal_flip', True) else A.NoOp(),
        A.VerticalFlip(p=0.5) if augmentation_config.get('train', {}).get('vertical_flip', False) else A.NoOp(),
        A.Rotate(
            limit=augmentation_config.get('train', {}).get('rotation_range', 10),
            p=0.5
        ) if augmentation_config.get('train', {}).get('rotation_range', 10) > 0 else A.NoOp(),
        A.ShiftScaleRotate(
            shift_limit=augmentation_config.get('train', {}).get('width_shift_range', 0.1),
            scale_limit=augmentation_config.get('train', {}).get('zoom_range', 0.1),
            rotate_limit=0,
            p=0.5
        ),
        
        # Intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=augmentation_config.get('train', {}).get('brightness_range', [0.9, 1.1]),
            contrast_limit=augmentation_config.get('train', {}).get('contrast_range', [0.9, 1.1]),
            p=0.5
        ),
        A.GaussNoise(
            var_limit=augmentation_config.get('train', {}).get('noise_factor', 0.05),
            p=0.3
        ) if augmentation_config.get('train', {}).get('noise_factor', 0.05) > 0 else A.NoOp(),
        
        # Elastic transformations
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.3
        ) if augmentation_config.get('train', {}).get('elastic_transform', True) else A.NoOp(),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=0.3
        ) if augmentation_config.get('train', {}).get('grid_distortion', True) else A.NoOp(),
        A.OpticalDistortion(
            distort_limit=0.3,
            shift_limit=0.3,
            p=0.3
        ) if augmentation_config.get('train', {}).get('optical_distortion', True) else A.NoOp(),
        
        # Normalization and conversion
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    # Validation transforms (minimal)
    val_transforms = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    # Test transforms (same as validation)
    test_transforms = val_transforms
    
    return train_transforms, val_transforms, test_transforms

def get_torchvision_transforms(image_size: Tuple[int, int] = (224, 224), 
                             augmentation: bool = True):
    """
    Get torchvision transforms (alternative to albumentations)
    
    Args:
        image_size: Target image size
        augmentation: Whether to apply augmentation
    
    Returns:
        Tuple of (train_transforms, val_transforms, test_transforms)
    """
    # Base transforms
    base_transforms = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    # Training transforms with augmentation
    if augmentation:
        train_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transforms = transforms.Compose(base_transforms)
    
    # Validation and test transforms
    val_transforms = transforms.Compose(base_transforms)
    test_transforms = transforms.Compose(base_transforms)
    
    return train_transforms, val_transforms, test_transforms

def get_medical_transforms(image_size: Tuple[int, int] = (224, 224)):
    """
    Get medical imaging specific transforms
    
    Args:
        image_size: Target image size
    
    Returns:
        Tuple of (train_transforms, val_transforms, test_transforms)
    """
    # Medical imaging specific augmentations
    train_transforms = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        
        # Medical imaging specific augmentations
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.5
        ),
        
        # Intensity augmentations (conservative for medical images)
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5
        ),
        A.GaussNoise(var_limit=0.02, p=0.3),
        
        # Medical imaging specific
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomGamma(gamma_limit=(0.8, 1.2), p=0.3),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    val_transforms = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    test_transforms = val_transforms
    
    return train_transforms, val_transforms, test_transforms

def get_strong_augmentation_transforms(image_size: Tuple[int, int] = (224, 224)):
    """
    Get strong augmentation transforms for data augmentation
    
    Args:
        image_size: Target image size
    
    Returns:
        Tuple of (train_transforms, val_transforms, test_transforms)
    """
    train_transforms = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        
        # Strong geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.7),
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=20,
            p=0.7
        ),
        
        # Strong intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.GaussNoise(var_limit=0.1, p=0.5),
        A.MultiplicativeNoise(multiplier=[0.8, 1.2], p=0.3),
        
        # Advanced augmentations
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.5, p=0.5),
        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        
        # Color augmentations
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    val_transforms = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    test_transforms = val_transforms
    
    return train_transforms, val_transforms, test_transforms

def get_test_time_augmentation_transforms(image_size: Tuple[int, int] = (224, 224)):
    """
    Get test time augmentation transforms
    
    Args:
        image_size: Target image size
    
    Returns:
        List of transforms for TTA
    """
    tta_transforms = [
        # Original
        A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Horizontal flip
        A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Rotation 90 degrees
        A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Rotate(limit=90, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Rotation -90 degrees
        A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Rotate(limit=-90, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Slight brightness increase
        A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Slight brightness decrease
        A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.RandomBrightnessContrast(brightness_limit=-0.1, contrast_limit=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    ]
    
    return tta_transforms

def apply_transform_to_image(image: np.ndarray, transform) -> torch.Tensor:
    """
    Apply transform to a single image
    
    Args:
        image: Input image as numpy array
        transform: Transform to apply
    
    Returns:
        Transformed image as tensor
    """
    if isinstance(transform, A.Compose):
        # Albumentations transform
        transformed = transform(image=image)
        return transformed['image']
    else:
        # Torchvision transform
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return transform(image)

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize image tensor back to [0, 255] range
    
    Args:
        tensor: Normalized image tensor
    
    Returns:
        Denormalized image as numpy array
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    denormalized = tensor * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)
    
    # Convert to numpy and scale to [0, 255]
    denormalized = denormalized.permute(1, 2, 0).numpy()
    denormalized = (denormalized * 255).astype(np.uint8)
    
    return denormalized

def create_mixup_batch(images: torch.Tensor, labels: torch.Tensor, 
                      alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Create mixup batch for data augmentation
    
    Args:
        images: Batch of images
        labels: Batch of labels
        alpha: Mixup parameter
    
    Returns:
        Tuple of (mixed_images, labels_a, labels_b, lam)
    """
    batch_size = images.size(0)
    
    # Sample lambda from beta distribution
    lam = np.random.beta(alpha, alpha)
    
    # Create random permutation
    index = torch.randperm(batch_size)
    
    # Mix images
    mixed_images = lam * images + (1 - lam) * images[index, :]
    
    return mixed_images, labels, labels[index], lam

def create_cutmix_batch(images: torch.Tensor, labels: torch.Tensor, 
                       alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Create cutmix batch for data augmentation
    
    Args:
        images: Batch of images
        labels: Batch of labels
        alpha: Cutmix parameter
    
    Returns:
        Tuple of (mixed_images, labels_a, labels_b, lam)
    """
    batch_size = images.size(0)
    
    # Sample lambda from beta distribution
    lam = np.random.beta(alpha, alpha)
    
    # Create random permutation
    index = torch.randperm(batch_size)
    
    # Get image dimensions
    h, w = images.size(2), images.size(3)
    
    # Calculate cutmix box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    # Random center
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    # Define box boundaries
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Apply cutmix
    mixed_images = images.clone()
    mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    return mixed_images, labels, labels[index], lam