#!/usr/bin/env python3
"""
Main training script for Pneumonia Detection with Explainable AI
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import *
from utils.data_loader import ChestXRayDataset
from utils.models import get_model
from utils.transforms import get_transforms
from utils.metrics import calculate_metrics
from utils.visualization import plot_training_history, plot_confusion_matrix
from utils.logging_utils import setup_logging

class PneumoniaTrainer:
    """Main trainer class for pneumonia detection models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        self.logger = setup_logging()
        
        # Initialize components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def _setup_device(self) -> torch.device:
        """Setup device (GPU/CPU) for training"""
        if self.config['hardware']['device'] == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.config['hardware']['device'])
        
        self.logger.info(f"Using device: {device}")
        return device
    
    def setup_data(self, dataset_name: str = 'chest_xray'):
        """Setup data loaders"""
        dataset_config = DATASETS[dataset_name]
        
        # Get transforms
        train_transforms, val_transforms, test_transforms = get_transforms(
            dataset_config['image_size'],
            AUGMENTATION_CONFIG
        )
        
        # Create datasets
        train_dataset = ChestXRayDataset(
            data_dir=DATA_DIR / dataset_name / 'train',
            transform=train_transforms,
            classes=dataset_config['classes']
        )
        
        val_dataset = ChestXRayDataset(
            data_dir=DATA_DIR / dataset_name / 'val',
            transform=val_transforms,
            classes=dataset_config['classes']
        )
        
        test_dataset = ChestXRayDataset(
            data_dir=DATA_DIR / dataset_name / 'test',
            transform=test_transforms,
            classes=dataset_config['classes']
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=TRAINING_CONFIG['batch_size'],
            shuffle=True,
            num_workers=HARDWARE_CONFIG['num_workers'],
            pin_memory=HARDWARE_CONFIG['pin_memory']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=TRAINING_CONFIG['batch_size'],
            shuffle=False,
            num_workers=HARDWARE_CONFIG['num_workers'],
            pin_memory=HARDWARE_CONFIG['pin_memory']
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=TRAINING_CONFIG['batch_size'],
            shuffle=False,
            num_workers=HARDWARE_CONFIG['num_workers'],
            pin_memory=HARDWARE_CONFIG['pin_memory']
        )
        
        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        self.logger.info(f"Test samples: {len(test_dataset)}")
    
    def setup_model(self, model_name: str):
        """Setup model, criterion, optimizer, and scheduler"""
        model_config = MODEL_CONFIG[model_name]
        
        # Get model
        self.model = get_model(model_name, model_config)
        self.model = self.model.to(self.device)
        
        # Setup criterion
        if TRAINING_CONFIG['class_weights']:
            # Calculate class weights
            class_counts = self._get_class_counts()
            class_weights = torch.FloatTensor([
                1.0 / class_counts[i] for i in range(len(class_counts))
            ]).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        if model_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=model_config['learning_rate'],
                weight_decay=model_config['weight_decay']
            )
        elif model_config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=model_config['learning_rate'],
                momentum=0.9,
                weight_decay=model_config['weight_decay']
            )
        
        # Setup scheduler
        if model_config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=TRAINING_CONFIG['epochs']
            )
        elif model_config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        
        self.logger.info(f"Model: {model_config['name']}")
        self.logger.info(f"Optimizer: {model_config['optimizer']}")
        self.logger.info(f"Scheduler: {model_config['scheduler']}")
    
    def _get_class_counts(self) -> np.ndarray:
        """Get class counts for calculating class weights"""
        class_counts = np.zeros(len(DATASETS['chest_xray']['classes']))
        
        for batch_idx, (_, labels) in enumerate(self.train_loader):
            for label in labels:
                class_counts[label] += 1
        
        return class_counts
    
    def train_epoch(self) -> tuple:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if TRAINING_CONFIG['gradient_clipping'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    TRAINING_CONFIG['gradient_clipping']
                )
            
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                self.logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                               f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> tuple:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs: int = None):
        """Main training loop"""
        if epochs is None:
            epochs = TRAINING_CONFIG['epochs']
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.info("Starting training...")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log progress
            self.logger.info(f'Epoch {epoch+1}/{epochs}:')
            self.logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            self.logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if TRAINING_CONFIG['early_stopping']:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model('best')
                else:
                    patience_counter += 1
                    if patience_counter >= TRAINING_CONFIG['patience']:
                        self.logger.info(f'Early stopping at epoch {epoch+1}')
                        break
        
        # Save final model
        self.save_model('final')
        
        # Plot training history
        self.plot_training_history()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        # Log results
        self.logger.info("Test Results:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(all_targets, all_predictions)
        
        return metrics
    
    def save_model(self, suffix: str):
        """Save model checkpoint"""
        model_path = MODELS_DIR / f"{self.config['model_name']}_{suffix}.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, model_path)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / f"{self.config['model_name']}_training_history.png")
        plt.close()
    
    def plot_confusion_matrix(self, targets, predictions):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=DATASETS['chest_xray']['classes'],
                   yticklabels=DATASETS['chest_xray']['classes'])
        plt.title(f'Confusion Matrix - {self.config["model_name"]}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / f"{self.config['model_name']}_confusion_matrix.png")
        plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train pneumonia detection model')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=list(MODEL_CONFIG.keys()),
                       help='Model architecture to train')
    parser.add_argument('--dataset', type=str, default='chest_xray',
                       choices=list(DATASETS.keys()),
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = {
        'model_name': args.model,
        'dataset_name': args.dataset,
        'hardware': HARDWARE_CONFIG,
        'training': TRAINING_CONFIG.copy(),
        'model': MODEL_CONFIG[args.model].copy()
    }
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['model']['learning_rate'] = args.learning_rate
    if args.output_dir:
        global OUTPUTS_DIR
        OUTPUTS_DIR = Path(args.output_dir)
        OUTPUTS_DIR.mkdir(exist_ok=True)
    
    # Initialize trainer
    trainer = PneumoniaTrainer(config)
    
    # Setup data and model
    trainer.setup_data(args.dataset)
    trainer.setup_model(args.model)
    
    # Train model
    trainer.train(config['training']['epochs'])
    
    # Evaluate model
    metrics = trainer.evaluate()
    
    # Save results
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(OUTPUTS_DIR / f"{args.model}_results.csv", index=False)
    
    print(f"Training completed! Results saved to {OUTPUTS_DIR}")

if __name__ == '__main__':
    main()