"""
Data loader utilities for chest X-ray datasets
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging

class ChestXRayDataset(Dataset):
    """Dataset class for chest X-ray images"""
    
    def __init__(self, data_dir: Path, transform=None, classes: List[str] = None, 
                 mode: str = 'train'):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing the dataset
            transform: Image transformations
            classes: List of class names
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = classes or ['NORMAL', 'PNEUMONIA']
        self.mode = mode
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load data
        self.samples = self._load_samples()
        
        logging.info(f"Loaded {len(self.samples)} samples for {mode} set")
        logging.info(f"Class distribution: {self._get_class_distribution()}")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load sample paths and labels"""
        samples = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = [
                f for f in class_dir.iterdir() 
                if f.suffix.lower() in image_extensions
            ]
            
            # Add samples
            for image_file in image_files:
                samples.append((str(image_file), self.class_to_idx[class_name]))
        
        return samples
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution"""
        distribution = {}
        for _, label in self.samples:
            class_name = self.classes[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample"""
        image_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

class PneumoniaSubtypeDataset(ChestXRayDataset):
    """Dataset for pneumonia subtypes (bacterial vs viral)"""
    
    def __init__(self, data_dir: Path, transform=None, mode: str = 'train'):
        """
        Initialize dataset with pneumonia subtypes
        
        Args:
            data_dir: Directory containing the dataset
            transform: Image transformations
            mode: Dataset mode ('train', 'val', 'test')
        """
        # Define classes for pneumonia subtypes
        classes = ['NORMAL', 'BACTERIA', 'VIRUS']
        
        super().__init__(data_dir, transform, classes, mode)
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load sample paths and labels with pneumonia subtypes"""
        samples = []
        
        # Handle normal class
        normal_dir = self.data_dir / 'NORMAL'
        if normal_dir.exists():
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            normal_files = [
                f for f in normal_dir.iterdir() 
                if f.suffix.lower() in image_extensions
            ]
            for image_file in normal_files:
                samples.append((str(image_file), 0))  # Normal = 0
        
        # Handle pneumonia classes
        pneumonia_dir = self.data_dir / 'PNEUMONIA'
        if pneumonia_dir.exists():
            # Bacterial pneumonia
            bacteria_dir = pneumonia_dir / 'BACTERIA'
            if bacteria_dir.exists():
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
                bacteria_files = [
                    f for f in bacteria_dir.iterdir() 
                    if f.suffix.lower() in image_extensions
                ]
                for image_file in bacteria_files:
                    samples.append((str(image_file), 1))  # Bacteria = 1
            
            # Viral pneumonia
            virus_dir = pneumonia_dir / 'VIRUS'
            if virus_dir.exists():
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
                virus_files = [
                    f for f in virus_dir.iterdir() 
                    if f.suffix.lower() in image_extensions
                ]
                for image_file in virus_files:
                    samples.append((str(image_file), 2))  # Virus = 2
        
        return samples

class NIHCheXpertDataset(Dataset):
    """Dataset for NIH Chest X-ray14 and CheXpert datasets"""
    
    def __init__(self, data_dir: Path, csv_file: Path, transform=None, 
                 target_columns: List[str] = None, mode: str = 'train'):
        """
        Initialize NIH/CheXpert dataset
        
        Args:
            data_dir: Directory containing images
            csv_file: Path to CSV file with labels
            transform: Image transformations
            target_columns: Columns to use as targets
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.data_dir = Path(data_dir)
        self.csv_file = Path(csv_file)
        self.transform = transform
        self.target_columns = target_columns or ['Pneumonia']
        self.mode = mode
        
        # Load CSV data
        self.data = pd.read_csv(self.csv_file)
        
        # Filter data based on mode
        if 'split' in self.data.columns:
            self.data = self.data[self.data['split'] == mode]
        
        logging.info(f"Loaded {len(self.data)} samples for {mode} set")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample"""
        row = self.data.iloc[idx]
        
        # Load image
        image_path = self.data_dir / row['Image_Index']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Get labels
        labels = []
        for col in self.target_columns:
            if col in row:
                # Handle different label formats
                value = row[col]
                if pd.isna(value) or value == -1:
                    labels.append(0)  # Uncertain/negative
                elif value == 1:
                    labels.append(1)  # Positive
                else:
                    labels.append(0)  # Negative
            else:
                labels.append(0)
        
        labels = torch.FloatTensor(labels)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, labels

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
        A.HorizontalFlip(p=0.5) if augmentation_config['train']['horizontal_flip'] else A.NoOp(),
        A.VerticalFlip(p=0.5) if augmentation_config['train']['vertical_flip'] else A.NoOp(),
        A.Rotate(
            limit=augmentation_config['train']['rotation_range'],
            p=0.5
        ) if augmentation_config['train']['rotation_range'] > 0 else A.NoOp(),
        A.ShiftScaleRotate(
            shift_limit=augmentation_config['train']['width_shift_range'],
            scale_limit=augmentation_config['train']['zoom_range'],
            rotate_limit=0,
            p=0.5
        ),
        
        # Intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=augmentation_config['train']['brightness_range'],
            contrast_limit=augmentation_config['train']['contrast_range'],
            p=0.5
        ),
        A.GaussNoise(
            var_limit=augmentation_config['train']['noise_factor'],
            p=0.3
        ) if augmentation_config['train']['noise_factor'] > 0 else A.NoOp(),
        
        # Elastic transformations
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.3
        ) if augmentation_config['train']['elastic_transform'] else A.NoOp(),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=0.3
        ) if augmentation_config['train']['grid_distortion'] else A.NoOp(),
        A.OpticalDistortion(
            distort_limit=0.3,
            shift_limit=0.3,
            p=0.3
        ) if augmentation_config['train']['optical_distortion'] else A.NoOp(),
        
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

def create_data_loaders(data_dir: Path, batch_size: int = 32, 
                       num_workers: int = 4, image_size: Tuple[int, int] = (224, 224),
                       augmentation_config: Dict[str, Any] = None,
                       dataset_type: str = 'chest_xray') -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and test sets
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        image_size: Target image size
        augmentation_config: Augmentation configuration
        dataset_type: Type of dataset ('chest_xray', 'nih', 'chexpert')
    
    Returns:
        Dictionary containing train, val, and test data loaders
    """
    # Get transforms
    train_transforms, val_transforms, test_transforms = get_transforms(
        image_size, augmentation_config or {}
    )
    
    # Create datasets based on type
    if dataset_type == 'chest_xray':
        train_dataset = ChestXRayDataset(
            data_dir / 'train',
            transform=train_transforms,
            mode='train'
        )
        val_dataset = ChestXRayDataset(
            data_dir / 'val',
            transform=val_transforms,
            mode='val'
        )
        test_dataset = ChestXRayDataset(
            data_dir / 'test',
            transform=test_transforms,
            mode='test'
        )
    
    elif dataset_type == 'pneumonia_subtypes':
        train_dataset = PneumoniaSubtypeDataset(
            data_dir / 'train',
            transform=train_transforms,
            mode='train'
        )
        val_dataset = PneumoniaSubtypeDataset(
            data_dir / 'val',
            transform=val_transforms,
            mode='val'
        )
        test_dataset = PneumoniaSubtypeDataset(
            data_dir / 'test',
            transform=test_transforms,
            mode='test'
        )
    
    elif dataset_type in ['nih', 'chexpert']:
        # For NIH and CheXpert, we need CSV files
        train_dataset = NIHCheXpertDataset(
            data_dir,
            csv_file=data_dir / f'{dataset_type}_train.csv',
            transform=train_transforms,
            mode='train'
        )
        val_dataset = NIHCheXpertDataset(
            data_dir,
            csv_file=data_dir / f'{dataset_type}_val.csv',
            transform=val_transforms,
            mode='val'
        )
        test_dataset = NIHCheXpertDataset(
            data_dir,
            csv_file=data_dir / f'{dataset_type}_test.csv',
            transform=test_transforms,
            mode='test'
        )
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }

def split_dataset(data_dir: Path, train_ratio: float = 0.8, 
                 val_ratio: float = 0.1, test_ratio: float = 0.1,
                 random_state: int = 42):
    """
    Split dataset into train/val/test sets
    
    Args:
        data_dir: Directory containing the dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    data_dir = Path(data_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in ['NORMAL', 'PNEUMONIA']:
            (data_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in class_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        # Shuffle files
        np.random.seed(random_state)
        np.random.shuffle(image_files)
        
        # Calculate split indices
        n_files = len(image_files)
        train_end = int(n_files * train_ratio)
        val_end = train_end + int(n_files * val_ratio)
        
        # Split files
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # Move files to appropriate directories
        for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
            for file_path in files:
                # Create symbolic link or copy file
                target_path = data_dir / split / class_name / file_path.name
                if not target_path.exists():
                    # Use symbolic link to save space
                    try:
                        target_path.symlink_to(file_path)
                    except OSError:
                        # Fallback to copying if symbolic link fails
                        import shutil
                        shutil.copy2(file_path, target_path)
    
    logging.info(f"Dataset split completed. Files moved to train/val/test directories.")