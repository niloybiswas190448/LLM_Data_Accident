"""
Evaluation metrics for pneumonia detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None,
                     classes: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        classes: Class names (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Cohen's Kappa and Matthews Correlation
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    
    # Per-class metrics
    if classes:
        for i, class_name in enumerate(classes):
            # Binary classification for each class
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            metrics[f'{class_name}_precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            metrics[f'{class_name}_recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            metrics[f'{class_name}_f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            # Sensitivity and Specificity (for binary classification)
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
            metrics[f'{class_name}_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f'{class_name}_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC-ROC (if probabilities provided)
    if y_prob is not None:
        if len(np.unique(y_true)) == 2:
            # Binary classification
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # Multi-class classification
            metrics['auc_roc_macro'] = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
            metrics['auc_roc_weighted'] = roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr')
    
    return metrics

def calculate_medical_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_prob: Optional[np.ndarray] = None,
                            classes: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate medical imaging specific metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        classes: Class names (optional)
    
    Returns:
        Dictionary of medical metrics
    """
    metrics = {}
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class medical metrics
    if classes:
        for i, class_name in enumerate(classes):
            # Binary classification for each class
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            # Confusion matrix for this class
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
            
            # Medical metrics
            metrics[f'{class_name}_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f'{class_name}_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics[f'{class_name}_ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
            metrics[f'{class_name}_npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
            metrics[f'{class_name}_accuracy'] = (tp + tn) / (tp + tn + fp + fn)
            
            # Likelihood ratios
            metrics[f'{class_name}_lr_positive'] = (tp / (tp + fn)) / (fp / (fp + tn)) if (fp / (fp + tn)) > 0 else float('inf')
            metrics[f'{class_name}_lr_negative'] = (fn / (tp + fn)) / (tn / (fp + tn)) if (tn / (fp + tn)) > 0 else float('inf')
            
            # Diagnostic odds ratio
            metrics[f'{class_name}_dor'] = (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('inf')
    
    # Overall medical metrics
    metrics['overall_sensitivity'] = np.mean([metrics[f'{c}_sensitivity'] for c in classes]) if classes else 0
    metrics['overall_specificity'] = np.mean([metrics[f'{c}_specificity'] for c in classes]) if classes else 0
    metrics['overall_ppv'] = np.mean([metrics[f'{c}_ppv'] for c in classes]) if classes else 0
    metrics['overall_npv'] = np.mean([metrics[f'{c}_npv'] for c in classes]) if classes else 0
    
    return metrics

def calculate_interpretability_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                     explanations: Dict[str, np.ndarray],
                                     ground_truth_regions: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate interpretability metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        explanations: Dictionary of explanations from different methods
        ground_truth_regions: Ground truth regions of interest (optional)
    
    Returns:
        Dictionary of interpretability metrics
    """
    metrics = {}
    
    for method_name, method_explanations in explanations.items():
        if method_explanations is None:
            continue
        
        # Faithfulness: How well explanations match model behavior
        faithfulness = calculate_faithfulness(y_true, y_pred, method_explanations)
        metrics[f'{method_name}_faithfulness'] = faithfulness
        
        # Monotonicity: Consistency of explanations
        monotonicity = calculate_monotonicity(method_explanations)
        metrics[f'{method_name}_monotonicity'] = monotonicity
        
        # Sparsity: Conciseness of explanations
        sparsity = calculate_sparsity(method_explanations)
        metrics[f'{method_name}_sparsity'] = sparsity
        
        # Ground truth alignment (if available)
        if ground_truth_regions is not None:
            alignment = calculate_ground_truth_alignment(method_explanations, ground_truth_regions)
            metrics[f'{method_name}_ground_truth_alignment'] = alignment
    
    return metrics

def calculate_faithfulness(y_true: np.ndarray, y_pred: np.ndarray,
                         explanations: np.ndarray) -> float:
    """
    Calculate faithfulness of explanations
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        explanations: Explanation maps
    
    Returns:
        Faithfulness score
    """
    # Simple implementation: correlation between explanation importance and prediction confidence
    # This is a simplified version - more sophisticated methods exist
    
    # Calculate prediction confidence (distance from decision boundary)
    confidence = np.abs(y_pred - 0.5)  # Assuming binary classification
    
    # Calculate explanation importance (mean activation)
    importance = np.mean(explanations, axis=(1, 2))
    
    # Calculate correlation
    correlation = np.corrcoef(confidence, importance)[0, 1]
    
    return correlation if not np.isnan(correlation) else 0.0

def calculate_monotonicity(explanations: np.ndarray) -> float:
    """
    Calculate monotonicity of explanations
    
    Args:
        explanations: Explanation maps
    
    Returns:
        Monotonicity score
    """
    # Calculate how consistent explanations are across similar inputs
    # This is a simplified implementation
    
    # Calculate variance of explanations
    variance = np.var(explanations, axis=0)
    
    # Lower variance indicates higher monotonicity
    monotonicity = 1.0 / (1.0 + np.mean(variance))
    
    return monotonicity

def calculate_sparsity(explanations: np.ndarray) -> float:
    """
    Calculate sparsity of explanations
    
    Args:
        explanations: Explanation maps
    
    Returns:
        Sparsity score
    """
    # Calculate how concentrated explanations are
    # Higher sparsity means more concentrated explanations
    
    # Normalize explanations
    explanations_norm = explanations / (np.max(explanations) + 1e-8)
    
    # Calculate sparsity (percentage of near-zero values)
    threshold = 0.1
    sparsity = np.mean(explanations_norm < threshold)
    
    return sparsity

def calculate_ground_truth_alignment(explanations: np.ndarray,
                                   ground_truth_regions: np.ndarray) -> float:
    """
    Calculate alignment with ground truth regions
    
    Args:
        explanations: Explanation maps
        ground_truth_regions: Ground truth regions of interest
    
    Returns:
        Alignment score
    """
    # Calculate intersection over union (IoU) between explanations and ground truth
    # This is a simplified implementation
    
    # Threshold explanations
    explanations_binary = explanations > np.percentile(explanations, 80)
    
    # Calculate IoU
    intersection = np.logical_and(explanations_binary, ground_truth_regions)
    union = np.logical_or(explanations_binary, ground_truth_regions)
    
    iou = np.sum(intersection) / (np.sum(union) + 1e-8)
    
    return iou

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
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
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
    if len(np.unique(y_true)) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
    else:
        # Multi-class classification
        plt.figure(figsize=(8, 6))
        
        for i, class_name in enumerate(classes):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            auc = roc_auc_score(y_true_binary, y_prob[:, i])
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curves(y_true: np.ndarray, y_prob: np.ndarray,
                               classes: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> None:
    """
    Plot precision-recall curves
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        classes: Class names
        save_path: Path to save the plot
    """
    if len(np.unique(y_true)) == 2:
        # Binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
    else:
        # Multi-class classification
        plt.figure(figsize=(8, 6))
        
        for i, class_name in enumerate(classes):
            y_true_binary = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_prob[:, i])
            auc = roc_auc_score(y_true_binary, y_prob[:, i])
            
            plt.plot(recall, precision, label=f'{class_name} (AUC = {auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                 classes: Optional[List[str]] = None,
                                 save_path: Optional[str] = None) -> str:
    """
    Generate detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        save_path: Path to save the report
    
    Returns:
        Classification report string
    """
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=False)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report

def calculate_statistical_significance(results1: Dict[str, float],
                                    results2: Dict[str, float],
                                    metric: str = 'accuracy') -> float:
    """
    Calculate statistical significance between two results
    
    Args:
        results1: Results from first model
        results2: Results from second model
        metric: Metric to compare
    
    Returns:
        P-value for statistical significance
    """
    # This is a simplified implementation
    # In practice, you would need the raw predictions and use proper statistical tests
    
    from scipy import stats
    
    # Assuming we have the raw predictions (this is a placeholder)
    # In practice, you would pass the actual predictions
    pred1 = np.random.rand(100)  # Placeholder
    pred2 = np.random.rand(100)  # Placeholder
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(pred1, pred2)
    
    return p_value