"""
Configuration file for Pneumonia Detection with Explainable AI
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset configurations
DATASETS = {
    "chest_xray": {
        "name": "Chest X-Ray Images (Pneumonia)",
        "url": "https://storage.googleapis.com/kaggle-datasets-images/12/19/chest-xray-pneumonia.zip",
        "kaggle_dataset": "paultimothymooney/chest-xray-pneumonia",
        "classes": ["NORMAL", "PNEUMONIA"],
        "pneumonia_subtypes": ["BACTERIA", "VIRUS"],
        "image_size": (224, 224),
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "augmentation": True
    },
    "nih_cxr14": {
        "name": "NIH Chest X-ray14",
        "url": "https://storage.googleapis.com/nih-chest-xrays/images/images_001/images_001.tar.gz",
        "classes": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", 
                   "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", 
                   "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"],
        "image_size": (224, 224),
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "augmentation": True
    }
}

# Model configurations
MODEL_CONFIG = {
    "resnet50": {
        "name": "ResNet-50",
        "pretrained": True,
        "num_classes": 3,  # Normal, Bacterial Pneumonia, Viral Pneumonia
        "dropout": 0.5,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "scheduler": "cosine",
        "target_layer": "layer4",
        "input_size": (224, 224)
    },
    "resnet101": {
        "name": "ResNet-101",
        "pretrained": True,
        "num_classes": 3,
        "dropout": 0.5,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "scheduler": "cosine",
        "target_layer": "layer4",
        "input_size": (224, 224)
    },
    "densenet121": {
        "name": "DenseNet-121",
        "pretrained": True,
        "num_classes": 3,
        "dropout": 0.5,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "scheduler": "cosine",
        "target_layer": "denseblock4",
        "input_size": (224, 224)
    },
    "densenet169": {
        "name": "DenseNet-169",
        "pretrained": True,
        "num_classes": 3,
        "dropout": 0.5,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "scheduler": "cosine",
        "target_layer": "denseblock4",
        "input_size": (224, 224)
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "pretrained": True,
        "num_classes": 3,
        "dropout": 0.5,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "scheduler": "cosine",
        "target_layer": "features.7",
        "input_size": (224, 224)
    },
    "efficientnet_b4": {
        "name": "EfficientNet-B4",
        "pretrained": True,
        "num_classes": 3,
        "dropout": 0.5,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "scheduler": "cosine",
        "target_layer": "features.7",
        "input_size": (224, 224)
    },
    "vit_base": {
        "name": "Vision Transformer (Base)",
        "pretrained": True,
        "num_classes": 3,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "scheduler": "cosine",
        "patch_size": 16,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "input_size": (224, 224)
    }
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "early_stopping": True,
    "patience": 10,
    "min_delta": 1e-4,
    "cross_validation": True,
    "k_folds": 5,
    "mixed_precision": True,
    "gradient_clipping": 1.0,
    "class_weights": True,
    "focal_loss": False,
    "label_smoothing": 0.1
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    "train": {
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation_range": 10,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "zoom_range": 0.1,
        "brightness_range": [0.9, 1.1],
        "contrast_range": [0.9, 1.1],
        "noise_factor": 0.05,
        "elastic_transform": True,
        "grid_distortion": True,
        "optical_distortion": True
    },
    "val": {
        "horizontal_flip": False,
        "vertical_flip": False,
        "rotation_range": 0,
        "width_shift_range": 0,
        "height_shift_range": 0,
        "zoom_range": 0,
        "brightness_range": [1.0, 1.0],
        "contrast_range": [1.0, 1.0],
        "noise_factor": 0,
        "elastic_transform": False,
        "grid_distortion": False,
        "optical_distortion": False
    }
}

# Explainable AI configuration
XAI_CONFIG = {
    "methods": ["gradcam", "shap", "lime", "integrated_gradients", "attention_maps"],
    "target_layer": "layer4",
    "num_samples": 1000,
    "visualization": True,
    "save_explanations": True,
    "explanation_dir": OUTPUTS_DIR / "explanations",
    "gradcam": {
        "target_layer": "layer4",
        "colormap": "jet",
        "alpha": 0.4
    },
    "shap": {
        "background_samples": 100,
        "nsamples": 100,
        "l1_reg": "auto"
    },
    "lime": {
        "num_samples": 1000,
        "num_features": 10,
        "top_labels": 3
    },
    "integrated_gradients": {
        "steps": 50,
        "baseline": "black"
    },
    "attention_maps": {
        "head_attention": True,
        "layer_attention": True,
        "patch_attention": True
    }
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "auc_roc", "sensitivity", "specificity"],
    "confusion_matrix": True,
    "roc_curves": True,
    "pr_curves": True,
    "calibration_plots": True,
    "confidence_histograms": True,
    "statistical_tests": ["mcnemar", "wilcoxon", "friedman"],
    "cross_validation": True,
    "k_folds": 5,
    "stratified": True,
    "random_state": 42
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    "style": "seaborn-v0_8",
    "figsize": (12, 8),
    "dpi": 300,
    "save_format": "png",
    "color_palette": "viridis",
    "font_size": 12,
    "title_size": 14,
    "label_size": 10,
    "tick_size": 8,
    "grid": True,
    "transparent": False
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "pneumonia_detection.log",
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Hardware configuration
HARDWARE_CONFIG = {
    "device": "auto",  # "auto", "cuda", "cpu"
    "num_workers": 4,
    "pin_memory": True,
    "mixed_precision": True,
    "gradient_accumulation_steps": 1
}

# Clinical validation configuration
CLINICAL_CONFIG = {
    "radiologist_evaluation": True,
    "confidence_thresholds": [0.5, 0.7, 0.9],
    "uncertainty_quantification": True,
    "decision_support": True,
    "region_of_interest": True,
    "clinical_metrics": ["sensitivity", "specificity", "ppv", "npv", "accuracy"]
}

# Output configuration
OUTPUT_CONFIG = {
    "save_models": True,
    "save_predictions": True,
    "save_explanations": True,
    "save_visualizations": True,
    "save_reports": True,
    "compression": True,
    "overwrite": False
}

# API configuration (for web interface)
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "workers": 4,
    "timeout": 30,
    "max_file_size": 10 * 1024 * 1024  # 10MB
}

# Environment variables
ENV_VARS = {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4"
}

# Set environment variables
for key, value in ENV_VARS.items():
    if key not in os.environ:
        os.environ[key] = str(value)