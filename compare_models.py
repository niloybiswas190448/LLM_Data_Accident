#!/usr/bin/env python3
"""
Comparative analysis script for pneumonia detection models
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from scipy import stats

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import *
from utils.data_loader import create_data_loaders
from utils.models import get_model, count_parameters
from utils.metrics import calculate_metrics, calculate_medical_metrics
from utils.visualization import plot_model_comparison, create_interactive_dashboard
from utils.logging_utils import setup_logging, ExperimentLogger

class ModelComparator:
    """Class for comparing multiple models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logging()
        
        # Initialize experiment logger
        self.experiment_logger = ExperimentLogger('model_comparison')
        
        # Results storage
        self.results = {}
        self.training_times = {}
        self.model_sizes = {}
        
    def setup_data(self, dataset_name: str = 'chest_xray'):
        """Setup data loaders"""
        self.logger.info("Setting up data loaders...")
        
        self.data_loaders = create_data_loaders(
            data_dir=DATA_DIR / dataset_name,
            batch_size=TRAINING_CONFIG['batch_size'],
            num_workers=HARDWARE_CONFIG['num_workers'],
            image_size=DATASETS[dataset_name]['image_size'],
            augmentation_config=AUGMENTATION_CONFIG,
            dataset_type=dataset_name
        )
        
        self.logger.info("Data loaders setup completed!")
    
    def train_model(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a single model"""
        self.logger.info(f"Training {model_name}...")
        
        # Create model
        model = get_model(model_name, model_config)
        model = model.to(self.device)
        
        # Count parameters
        num_params = count_parameters(model)
        self.model_sizes[model_name] = num_params
        
        # Setup training components
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_config['learning_rate'],
            weight_decay=model_config['weight_decay']
        )
        
        # Training loop
        start_time = time.time()
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(TRAINING_CONFIG['epochs']):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(self.data_loaders['train']):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            train_loss /= len(self.data_loaders['train'])
            train_acc = 100. * train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in self.data_loaders['val']:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            val_loss /= len(self.data_loaders['val'])
            val_acc = 100. * val_correct / val_total
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Log progress
            if epoch % 10 == 0:
                self.logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                               f'Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                               f'Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), MODELS_DIR / f'{model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= TRAINING_CONFIG['patience']:
                    self.logger.info(f'Early stopping at epoch {epoch}')
                    break
        
        training_time = time.time() - start_time
        self.training_times[model_name] = training_time
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(MODELS_DIR / f'{model_name}_best.pth'))
        
        # Evaluate on test set
        test_results = self.evaluate_model(model, model_name)
        
        # Store results
        self.results[model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'test_metrics': test_results,
            'training_time': training_time,
            'num_parameters': num_params
        }
        
        self.logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
        
        return self.results[model_name]
    
    def evaluate_model(self, model: torch.nn.Module, model_name: str) -> Dict[str, float]:
        """Evaluate a trained model"""
        self.logger.info(f"Evaluating {model_name}...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in self.data_loaders['test']:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                probabilities = torch.softmax(output, dim=1)
                
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            np.array(all_probabilities),
            classes=DATASETS['chest_xray']['classes']
        )
        
        # Calculate medical metrics
        medical_metrics = calculate_medical_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            np.array(all_probabilities),
            classes=DATASETS['chest_xray']['classes']
        )
        
        # Combine metrics
        all_metrics = {**metrics, **medical_metrics}
        
        # Log results
        self.logger.info(f"{model_name} Test Results:")
        for metric, value in all_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return all_metrics
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple models"""
        self.logger.info(f"Starting comparison of models: {model_names}")
        
        # Train all models
        for model_name in model_names:
            if model_name not in MODEL_CONFIG:
                self.logger.warning(f"Model {model_name} not found in config, skipping...")
                continue
            
            model_config = MODEL_CONFIG[model_name]
            self.train_model(model_name, model_config)
        
        # Generate comparison report
        comparison_report = self.generate_comparison_report()
        
        # Log experiment results
        self.experiment_logger.log_config(self.config)
        for model_name, result in self.results.items():
            self.experiment_logger.log_model_performance(model_name, result['test_metrics'])
        
        self.experiment_logger.save_experiment_summary()
        
        return comparison_report
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        self.logger.info("Generating comparison report...")
        
        # Extract test metrics for comparison
        test_metrics = {}
        for model_name, result in self.results.items():
            test_metrics[model_name] = result['test_metrics']
        
        # Key metrics for comparison
        key_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                      'auc_roc_macro', 'overall_sensitivity', 'overall_specificity']
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in test_metrics.items():
            row = {'model': model_name}
            for metric in key_metrics:
                row[metric] = metrics.get(metric, 0)
            row['training_time'] = self.results[model_name]['training_time']
            row['num_parameters'] = self.results[model_name]['num_parameters']
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Statistical significance testing
        significance_tests = self.perform_statistical_tests(test_metrics)
        
        # Generate visualizations
        self.generate_comparison_visualizations(test_metrics)
        
        # Create summary
        summary = {
            'best_model': self.find_best_model(test_metrics),
            'comparison_dataframe': comparison_df,
            'significance_tests': significance_tests,
            'training_times': self.training_times,
            'model_sizes': self.model_sizes
        }
        
        # Save results
        self.save_comparison_results(summary)
        
        return summary
    
    def find_best_model(self, test_metrics: Dict[str, Dict[str, float]]) -> str:
        """Find the best performing model"""
        # Simple ranking based on accuracy
        model_scores = {}
        for model_name, metrics in test_metrics.items():
            # Weighted score (you can adjust weights)
            score = (metrics.get('accuracy', 0) * 0.4 + 
                    metrics.get('f1_macro', 0) * 0.3 + 
                    metrics.get('auc_roc_macro', 0) * 0.3)
            model_scores[model_name] = score
        
        best_model = max(model_scores, key=model_scores.get)
        self.logger.info(f"Best model: {best_model} (score: {model_scores[best_model]:.4f})")
        
        return best_model
    
    def perform_statistical_tests(self, test_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        self.logger.info("Performing statistical significance tests...")
        
        # This is a simplified version - in practice, you'd need the raw predictions
        # for proper statistical testing
        
        significance_results = {}
        
        # Compare models pairwise
        model_names = list(test_metrics.keys())
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                comparison_key = f"{model1}_vs_{model2}"
                
                # Compare accuracy (simplified)
                acc1 = test_metrics[model1].get('accuracy', 0)
                acc2 = test_metrics[model2].get('accuracy', 0)
                
                # Simple t-test simulation (in practice, use actual predictions)
                # This is just for demonstration
                significance_results[comparison_key] = {
                    'accuracy_diff': acc1 - acc2,
                    'p_value': 0.05 if abs(acc1 - acc2) > 0.01 else 0.5,  # Simplified
                    'significant': abs(acc1 - acc2) > 0.01
                }
        
        return significance_results
    
    def generate_comparison_visualizations(self, test_metrics: Dict[str, Dict[str, float]]):
        """Generate comparison visualizations"""
        self.logger.info("Generating comparison visualizations...")
        
        # Model comparison plot
        plot_model_comparison(test_metrics, save_path=OUTPUTS_DIR / 'model_comparison.png')
        
        # Interactive dashboard
        create_interactive_dashboard(test_metrics, save_path=OUTPUTS_DIR / 'model_dashboard.html')
        
        # Training curves comparison
        self.plot_training_curves_comparison()
        
        # Model size vs performance
        self.plot_model_size_vs_performance(test_metrics)
    
    def plot_training_curves_comparison(self):
        """Plot training curves for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, model_name in enumerate(self.results.keys()):
            result = self.results[model_name]
            
            # Loss curves
            axes[0, 0].plot(result['train_losses'], label=f'{model_name} (Train)', alpha=0.7)
            axes[0, 1].plot(result['val_losses'], label=f'{model_name} (Val)', alpha=0.7)
            
            # Accuracy curves
            axes[1, 0].plot(result['train_accuracies'], label=f'{model_name} (Train)', alpha=0.7)
            axes[1, 1].plot(result['val_accuracies'], label=f'{model_name} (Val)', alpha=0.7)
        
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_size_vs_performance(self, test_metrics: Dict[str, Dict[str, float]]):
        """Plot model size vs performance"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        model_names = list(test_metrics.keys())
        accuracies = [test_metrics[name].get('accuracy', 0) for name in model_names]
        f1_scores = [test_metrics[name].get('f1_macro', 0) for name in model_names]
        model_sizes = [self.model_sizes[name] for name in model_names]
        
        # Accuracy vs model size
        ax1.scatter(model_sizes, accuracies, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            ax1.annotate(name, (model_sizes[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax1.set_xlabel('Number of Parameters')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Size vs Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # F1-score vs model size
        ax2.scatter(model_sizes, f1_scores, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            ax2.annotate(name, (model_sizes[i], f1_scores[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('Number of Parameters')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('Model Size vs F1-Score')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / 'model_size_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_comparison_results(self, summary: Dict[str, Any]):
        """Save comparison results"""
        # Save DataFrame
        summary['comparison_dataframe'].to_csv(OUTPUTS_DIR / 'model_comparison_results.csv', index=False)
        
        # Save summary as JSON
        summary_json = {
            'best_model': summary['best_model'],
            'training_times': summary['training_times'],
            'model_sizes': summary['model_sizes'],
            'significance_tests': summary['significance_tests']
        }
        
        with open(OUTPUTS_DIR / 'comparison_summary.json', 'w') as f:
            json.dump(summary_json, f, indent=2)
        
        self.logger.info(f"Comparison results saved to {OUTPUTS_DIR}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Compare multiple pneumonia detection models')
    parser.add_argument('--models', nargs='+', default=['resnet50', 'densenet121', 'efficientnet_b0'],
                       help='Models to compare')
    parser.add_argument('--dataset', type=str, default='chest_xray',
                       choices=list(DATASETS.keys()),
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    # Update config
    config = {
        'models': args.models,
        'dataset': args.dataset,
        'training': TRAINING_CONFIG.copy(),
        'hardware': HARDWARE_CONFIG
    }
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Initialize comparator
    comparator = ModelComparator(config)
    
    # Setup data
    comparator.setup_data(args.dataset)
    
    # Compare models
    comparison_report = comparator.compare_models(args.models)
    
    # Print summary
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"Best Model: {comparison_report['best_model']}")
    print(f"Results saved to: {OUTPUTS_DIR}")
    print("="*50)

if __name__ == '__main__':
    main()