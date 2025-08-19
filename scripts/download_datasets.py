#!/usr/bin/env python3
"""
Dataset download and preparation script for pneumonia detection
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, DATASETS
from utils.logging_utils import setup_logging

def download_file(url: str, filepath: Path, chunk_size: int = 8192):
    """
    Download file with progress bar
    
    Args:
        url: URL to download from
        filepath: Path to save the file
        chunk_size: Chunk size for downloading
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            pbar.update(size)

def download_chest_xray_dataset():
    """Download Chest X-Ray Images (Pneumonia) dataset"""
    logger = setup_logging()
    logger.info("Downloading Chest X-Ray Images (Pneumonia) dataset...")
    
    dataset_config = DATASETS['chest_xray']
    dataset_dir = DATA_DIR / 'chest_xray'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Download from Kaggle (requires kaggle CLI)
    try:
        import kaggle
        logger.info("Using Kaggle API to download dataset...")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_config['kaggle_dataset'],
            path=str(dataset_dir),
            unzip=True
        )
        
        logger.info("Dataset downloaded successfully!")
        
    except ImportError:
        logger.warning("Kaggle API not available. Please install with: pip install kaggle")
        logger.info("Manual download instructions:")
        logger.info(f"1. Visit: https://www.kaggle.com/datasets/{dataset_config['kaggle_dataset']}")
        logger.info("2. Download the dataset")
        logger.info(f"3. Extract to: {dataset_dir}")
        
        # Alternative: Try direct download if available
        if dataset_config.get('url'):
            logger.info("Attempting direct download...")
            zip_path = dataset_dir / 'chest_xray_pneumonia.zip'
            
            try:
                download_file(dataset_config['url'], zip_path)
                
                # Extract
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                
                # Clean up
                zip_path.unlink()
                logger.info("Dataset downloaded and extracted successfully!")
                
            except Exception as e:
                logger.error(f"Direct download failed: {e}")
                logger.info("Please download manually from Kaggle")
    
    # Verify dataset structure
    verify_chest_xray_structure(dataset_dir)

def download_nih_dataset():
    """Download NIH Chest X-ray14 dataset"""
    logger = setup_logging()
    logger.info("Downloading NIH Chest X-ray14 dataset...")
    
    dataset_dir = DATA_DIR / 'nih_cxr14'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # NIH dataset URLs
    base_url = "https://storage.googleapis.com/nih-chest-xrays"
    
    # Download images
    images_url = f"{base_url}/images/images_001/images_001.tar.gz"
    images_path = dataset_dir / 'images_001.tar.gz'
    
    logger.info("Downloading images...")
    download_file(images_url, images_path)
    
    # Extract images
    logger.info("Extracting images...")
    with tarfile.open(images_path, 'r:gz') as tar:
        tar.extractall(dataset_dir)
    
    # Clean up
    images_path.unlink()
    
    # Download labels
    labels_url = "https://storage.googleapis.com/nih-chest-xrays/labels/Data_Entry_2017_v2020.csv"
    labels_path = dataset_dir / 'Data_Entry_2017_v2020.csv'
    
    logger.info("Downloading labels...")
    download_file(labels_url, labels_path)
    
    logger.info("NIH dataset downloaded successfully!")
    
    # Process labels
    process_nih_labels(labels_path)

def download_chexpert_dataset():
    """Download CheXpert dataset"""
    logger = setup_logging()
    logger.info("Downloading CheXpert dataset...")
    
    dataset_dir = DATA_DIR / 'chexpert'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # CheXpert requires registration and approval
    logger.info("CheXpert dataset requires registration and approval.")
    logger.info("Please visit: https://stanfordmlgroup.github.io/competitions/chexpert/")
    logger.info("After approval, download and extract to the chexpert directory.")
    
    # Check if dataset exists
    if (dataset_dir / 'train.csv').exists():
        logger.info("CheXpert dataset found!")
        process_chexpert_labels(dataset_dir)
    else:
        logger.warning("CheXpert dataset not found. Please download manually.")

def verify_chest_xray_structure(dataset_dir: Path):
    """Verify the structure of the chest X-ray dataset"""
    logger = setup_logging()
    
    expected_structure = {
        'train': ['NORMAL', 'PNEUMONIA'],
        'test': ['NORMAL', 'PNEUMONIA'],
        'val': ['NORMAL', 'PNEUMONIA']
    }
    
    logger.info("Verifying dataset structure...")
    
    for split, classes in expected_structure.items():
        split_dir = dataset_dir / split
        if not split_dir.exists():
            logger.warning(f"Missing {split} directory")
            continue
        
        for class_name in classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Missing {split}/{class_name} directory")
                continue
            
            # Count images
            image_files = list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.jpg'))
            logger.info(f"{split}/{class_name}: {len(image_files)} images")
    
    logger.info("Dataset structure verification completed!")

def process_nih_labels(labels_path: Path):
    """Process NIH dataset labels"""
    logger = setup_logging()
    logger.info("Processing NIH dataset labels...")
    
    # Read labels
    df = pd.read_csv(labels_path)
    
    # Create train/val/test splits
    np.random.seed(42)
    df['split'] = np.random.choice(['train', 'val', 'test'], size=len(df), p=[0.8, 0.1, 0.1])
    
    # Save splits
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        split_df.to_csv(labels_path.parent / f'nih_{split}.csv', index=False)
        logger.info(f"{split} split: {len(split_df)} samples")
    
    logger.info("NIH labels processed successfully!")

def process_chexpert_labels(dataset_dir: Path):
    """Process CheXpert dataset labels"""
    logger = setup_logging()
    logger.info("Processing CheXpert dataset labels...")
    
    # Read train labels
    train_df = pd.read_csv(dataset_dir / 'train.csv')
    val_df = pd.read_csv(dataset_dir / 'valid.csv')
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    
    # Combine and create test split
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Create test split from validation
    test_size = len(val_df) // 2
    test_indices = np.random.choice(val_df.index, size=test_size, replace=False)
    
    combined_df.loc[test_indices, 'split'] = 'test'
    
    # Save splits
    for split in ['train', 'val', 'test']:
        split_df = combined_df[combined_df['split'] == split]
        split_df.to_csv(dataset_dir / f'chexpert_{split}.csv', index=False)
        logger.info(f"{split} split: {len(split_df)} samples")
    
    logger.info("CheXpert labels processed successfully!")

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    logger = setup_logging()
    logger.info("Creating sample dataset...")
    
    sample_dir = DATA_DIR / 'sample'
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for class_name in ['NORMAL', 'PNEUMONIA']:
            (sample_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Create dummy images (you would replace this with actual images)
    from PIL import Image
    import numpy as np
    
    for split in ['train', 'val', 'test']:
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = sample_dir / split / class_name
            
            # Create 10 dummy images per class per split
            for i in range(10):
                # Create a simple dummy image
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(class_dir / f'sample_{i}.jpg')
    
    logger.info("Sample dataset created successfully!")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and prepare datasets')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['chest_xray', 'nih', 'chexpert', 'sample', 'all'],
                       help='Dataset to download')
    parser.add_argument('--force', action='store_true',
                       help='Force download even if dataset exists')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting dataset download and preparation...")
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == 'all' or args.dataset == 'chest_xray':
        if not (DATA_DIR / 'chest_xray').exists() or args.force:
            download_chest_xray_dataset()
        else:
            logger.info("Chest X-ray dataset already exists. Use --force to re-download.")
    
    if args.dataset == 'all' or args.dataset == 'nih':
        if not (DATA_DIR / 'nih_cxr14').exists() or args.force:
            download_nih_dataset()
        else:
            logger.info("NIH dataset already exists. Use --force to re-download.")
    
    if args.dataset == 'all' or args.dataset == 'chexpert':
        if not (DATA_DIR / 'chexpert').exists() or args.force:
            download_chexpert_dataset()
        else:
            logger.info("CheXpert dataset already exists. Use --force to re-download.")
    
    if args.dataset == 'sample':
        create_sample_dataset()
    
    logger.info("Dataset download and preparation completed!")

if __name__ == '__main__':
    main()