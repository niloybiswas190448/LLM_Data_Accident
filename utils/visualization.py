"""
Visualization utilities for pneumonia detection
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_training_history(train_losses: List[float], val_losses: List[float],
                         train_accuracies: List[float], val_accuracies: List[float],
                         save_path: Optional[str] = None) -> None:
    """
    Plot training history
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        train_accuracies: Training accuracies
        val_accuracies: Validation accuracies
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         classes: Optional[List[str]] = None,
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        save_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray,
                   classes: Optional[List[str]] = None,
                   save_path: Optional[str] = None) -> None:
    """
    Plot ROC curves
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        classes: Class names
        save_path: Path to save the plot
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    
    if len(np.unique(y_true)) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    else:
        # Multi-class classification
        plt.figure(figsize=(8, 6))
        
        for i, class_name in enumerate(classes):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            auc = roc_auc_score(y_true_binary, y_prob[:, i])
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_comparison(results: Dict[str, Dict[str, float]],
                         metrics: List[str] = None,
                         save_path: Optional[str] = None) -> None:
    """
    Plot model comparison
    
    Args:
        results: Dictionary of model results
        metrics: Metrics to compare
        save_path: Path to save the plot
    """
    if metrics is None:
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_roc_macro']
    
    # Prepare data
    model_names = list(results.keys())
    metric_values = {metric: [results[model].get(metric, 0) for model in model_names] 
                    for metric in metrics}
    
    # Create subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        values = metric_values[metric]
        
        bars = axes[i].bar(model_names, values, color=sns.color_palette("husl", len(model_names)))
        axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(metric.replace("_", " ").title(), fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_explanations(original_images: np.ndarray, explanations: Dict[str, np.ndarray],
                     class_names: List[str], save_path: Optional[str] = None) -> None:
    """
    Plot explanations from different XAI methods
    
    Args:
        original_images: Original images
        explanations: Dictionary of explanations
        class_names: Class names
        save_path: Path to save the plot
    """
    n_samples = len(original_images)
    n_methods = len(explanations)
    
    fig, axes = plt.subplots(n_samples, n_methods + 1, 
                            figsize=(4*(n_methods + 1), 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Original image
        axes[i, 0].imshow(original_images[i])
        axes[i, 0].set_title(f'Original Image {i+1}', fontsize=10)
        axes[i, 0].axis('off')
        
        # Explanations
        for j, (method_name, method_explanations) in enumerate(explanations.items()):
            if method_explanations[i] is not None:
                axes[i, j + 1].imshow(method_explanations[i], cmap='jet')
                axes[i, j + 1].set_title(f'{method_name.upper()}', fontsize=10)
                axes[i, j + 1].axis('off')
            else:
                axes[i, j + 1].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[i, j + 1].set_title(f'{method_name.upper()}', fontsize=10)
                axes[i, j + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(feature_importance: np.ndarray,
                          feature_names: List[str],
                          save_path: Optional[str] = None) -> None:
    """
    Plot feature importance
    
    Args:
        feature_importance: Feature importance scores
        feature_names: Feature names
        save_path: Path to save the plot
    """
    # Sort features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_importance = feature_importance[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(sorted_importance)), sorted_importance, 
                   color=sns.color_palette("husl", len(sorted_importance)))
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Feature Importance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, sorted_importance)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_attention_maps(images: np.ndarray, attention_maps: np.ndarray,
                       save_path: Optional[str] = None) -> None:
    """
    Plot attention maps
    
    Args:
        images: Original images
        attention_maps: Attention maps
        save_path: Path to save the plot
    """
    n_samples = len(images)
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f'Original Image {i+1}', fontsize=12)
        axes[i, 0].axis('off')
        
        # Attention map
        axes[i, 1].imshow(attention_maps[i], cmap='jet')
        axes[i, 1].set_title(f'Attention Map {i+1}', fontsize=12)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_uncertainty_analysis(predictions: np.ndarray, uncertainties: np.ndarray,
                            save_path: Optional[str] = None) -> None:
    """
    Plot uncertainty analysis
    
    Args:
        predictions: Model predictions
        uncertainties: Prediction uncertainties
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Uncertainty distribution
    ax1.hist(uncertainties, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Uncertainty', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Uncertainty Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Prediction vs Uncertainty
    ax2.scatter(predictions, uncertainties, alpha=0.6, color='red')
    ax2.set_xlabel('Prediction Confidence', fontsize=12)
    ax2.set_ylabel('Uncertainty', fontsize=12)
    ax2.set_title('Prediction vs Uncertainty', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_interactive_dashboard(results: Dict[str, Dict[str, float]],
                               save_path: Optional[str] = None) -> None:
    """
    Create interactive dashboard using Plotly
    
    Args:
        results: Dictionary of model results
        save_path: Path to save the HTML file
    """
    # Prepare data
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_roc_macro']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy Comparison', 'Precision Comparison', 
                       'Recall Comparison', 'F1-Score Comparison'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Add traces
    for i, metric in enumerate(metrics[:4]):
        values = [results[model].get(metric, 0) for model in model_names]
        
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        fig.add_trace(
            go.Bar(x=model_names, y=values, name=metric.replace('_', ' ').title()),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title_text="Model Performance Dashboard",
        showlegend=False,
        height=800
    )
    
    if save_path:
        fig.write_html(save_path)
    
    fig.show()

def plot_class_distribution(class_counts: Dict[str, int],
                          save_path: Optional[str] = None) -> None:
    """
    Plot class distribution
    
    Args:
        class_counts: Dictionary of class counts
        save_path: Path to save the plot
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color=sns.color_palette("husl", len(classes)))
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_learning_curves(train_sizes: np.ndarray, train_scores: np.ndarray,
                        val_scores: np.ndarray, save_path: Optional[str] = None) -> None:
    """
    Plot learning curves
    
    Args:
        train_sizes: Training set sizes
        train_scores: Training scores
        val_scores: Validation scores
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', 
             label='Training Score', linewidth=2, markersize=8)
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', 
             label='Cross-validation Score', linewidth=2, markersize=8)
    
    plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
    plt.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                     np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)
    
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Learning Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_calibration_curves(y_true: np.ndarray, y_prob: np.ndarray,
                           save_path: Optional[str] = None) -> None:
    """
    Plot calibration curves
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the plot
    """
    from sklearn.calibration import calibration_curve
    
    plt.figure(figsize=(8, 6))
    
    # Perfect calibration
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', alpha=0.5)
    
    # Model calibration
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob[:, 1], n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', 
             label='Model', linewidth=2, markersize=8)
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()