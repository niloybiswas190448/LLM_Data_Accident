#!/usr/bin/env python3
"""
Explainable AI script for Pneumonia Detection
Implements various XAI methods: Grad-CAM, SHAP, LIME, Integrated Gradients
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# XAI libraries
import shap
import lime
from lime import lime_image
import captum
from captum.attr import IntegratedGradients, Saliency, Occlusion

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import *
from utils.data_loader import ChestXRayDataset
from utils.models import get_model
from utils.transforms import get_transforms
from utils.visualization import plot_explanations
from utils.logging_utils import setup_logging

class PneumoniaExplainer:
    """Explainable AI class for pneumonia detection models"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logging()
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup data
        self.setup_data()
        
        # Initialize XAI methods
        self.xai_methods = {}
        self._setup_xai_methods()
        
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config from checkpoint
        model_config = checkpoint['config']['model']
        model_name = checkpoint['config']['model_name']
        
        # Create model
        model = get_model(model_name, model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        self.logger.info(f"Loaded model: {model_name}")
        return model
    
    def setup_data(self):
        """Setup data for explanations"""
        dataset_config = DATASETS['chest_xray']
        
        # Get transforms
        _, _, test_transforms = get_transforms(
            dataset_config['image_size'],
            AUGMENTATION_CONFIG
        )
        
        # Create test dataset
        self.test_dataset = ChestXRayDataset(
            data_dir=DATA_DIR / 'chest_xray' / 'test',
            transform=test_transforms,
            classes=dataset_config['classes']
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,  # Process one image at a time for explanations
            shuffle=False,
            num_workers=1
        )
        
        self.logger.info(f"Test samples: {len(self.test_dataset)}")
    
    def _setup_xai_methods(self):
        """Setup XAI methods"""
        # Grad-CAM
        if 'gradcam' in XAI_CONFIG['methods']:
            self.xai_methods['gradcam'] = self._gradcam_explanation
        
        # SHAP
        if 'shap' in XAI_CONFIG['methods']:
            self.xai_methods['shap'] = self._shap_explanation
        
        # LIME
        if 'lime' in XAI_CONFIG['methods']:
            self.xai_methods['lime'] = self._lime_explanation
        
        # Integrated Gradients
        if 'integrated_gradients' in XAI_CONFIG['methods']:
            self.xai_methods['integrated_gradients'] = self._integrated_gradients_explanation
        
        # Saliency Maps
        if 'saliency' in XAI_CONFIG['methods']:
            self.xai_methods['saliency'] = self._saliency_explanation
        
        # Occlusion
        if 'occlusion' in XAI_CONFIG['methods']:
            self.xai_methods['occlusion'] = self._occlusion_explanation
    
    def _gradcam_explanation(self, image: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate Grad-CAM explanation"""
        # Get target layer
        target_layer = self._get_target_layer()
        
        # Register hook to get gradients and activations
        gradients = []
        activations = []
        
        def save_gradient(grad):
            gradients.append(grad)
        
        def save_activation(module, input, output):
            activations.append(output)
        
        # Register hooks
        handle_forward = target_layer.register_forward_hook(save_activation)
        handle_backward = target_layer.register_backward_hook(
            lambda module, grad_input, grad_output: save_gradient(grad_output[0])
        )
        
        # Forward pass
        image.requires_grad_(True)
        output = self.model(image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = gradients[0]
        activations = activations[0]
        
        # Calculate weights
        weights = torch.mean(gradients, dim=[2, 3])
        
        # Generate CAM
        cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * activations, dim=1)
        cam = F.relu(cam)  # Apply ReLU
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(cam.unsqueeze(0), size=image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze(0).squeeze(0)
        
        # Remove hooks
        handle_forward.remove()
        handle_backward.remove()
        
        return cam.detach().cpu().numpy()
    
    def _get_target_layer(self):
        """Get target layer for Grad-CAM"""
        model_name = self.config.get('model_name', 'resnet50')
        target_layer_name = MODEL_CONFIG[model_name]['target_layer']
        
        if model_name.startswith('resnet'):
            return getattr(self.model, target_layer_name)
        elif model_name.startswith('densenet'):
            return getattr(self.model.features, target_layer_name)
        elif model_name.startswith('efficientnet'):
            return getattr(self.model.features, target_layer_name)
        else:
            # Default to last layer
            return list(self.model.children())[-1]
    
    def _shap_explanation(self, image: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate SHAP explanation"""
        # Convert to numpy for SHAP
        image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Create background dataset
        background_images = []
        for i, (bg_image, _) in enumerate(self.test_loader):
            if i >= XAI_CONFIG['shap']['background_samples']:
                break
            bg_np = bg_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            background_images.append(bg_np)
        
        background_images = np.array(background_images)
        
        # Create SHAP explainer
        def model_predict(images):
            # Convert numpy to tensor
            images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(images_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            return probabilities.cpu().numpy()
        
        explainer = shap.KernelExplainer(model_predict, background_images)
        
        # Generate explanation
        shap_values = explainer.shap_values(
            image_np.reshape(1, -1),
            nsamples=XAI_CONFIG['shap']['nsamples']
        )
        
        # Reshape to image format
        if target_class is None:
            # Use the class with highest SHAP values
            target_class = np.argmax(np.sum(shap_values, axis=(1, 2)))
        
        explanation = shap_values[target_class].reshape(image_np.shape[:2])
        
        return explanation
    
    def _lime_explanation(self, image: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate LIME explanation"""
        # Convert to numpy for LIME
        image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Define prediction function
        def model_predict(images):
            # Convert numpy to tensor
            images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(images_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            return probabilities.cpu().numpy()
        
        # Generate explanation
        explanation = explainer.explain_instance(
            image_np,
            model_predict,
            top_labels=XAI_CONFIG['lime']['top_labels'],
            num_samples=XAI_CONFIG['lime']['num_samples'],
            num_features=XAI_CONFIG['lime']['num_features']
        )
        
        # Get explanation for target class
        if target_class is None:
            target_class = explanation.top_labels[0]
        
        # Get image and mask
        image, mask = explanation.get_image_and_mask(
            target_class,
            positive_only=False,
            num_features=XAI_CONFIG['lime']['num_features'],
            hide_rest=False
        )
        
        return mask
    
    def _integrated_gradients_explanation(self, image: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate Integrated Gradients explanation"""
        # Create Integrated Gradients explainer
        ig = IntegratedGradients(self.model)
        
        # Get target class
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = output.argmax(dim=1).item()
        
        # Create baseline (black image)
        baseline = torch.zeros_like(image)
        
        # Generate attribution
        attribution = ig.attribute(
            image,
            baseline,
            target=target_class,
            n_steps=XAI_CONFIG['integrated_gradients']['steps']
        )
        
        # Convert to numpy and normalize
        attribution = attribution.squeeze(0).permute(1, 2, 0).cpu().numpy()
        attribution = np.abs(attribution).sum(axis=2)  # Sum across channels
        
        # Normalize
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        return attribution
    
    def _saliency_explanation(self, image: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate Saliency Map explanation"""
        # Create Saliency explainer
        saliency = Saliency(self.model)
        
        # Get target class
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = output.argmax(dim=1).item()
        
        # Generate attribution
        attribution = saliency.attribute(image, target=target_class)
        
        # Convert to numpy and normalize
        attribution = attribution.squeeze(0).permute(1, 2, 0).cpu().numpy()
        attribution = np.abs(attribution).sum(axis=2)  # Sum across channels
        
        # Normalize
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        return attribution
    
    def _occlusion_explanation(self, image: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate Occlusion explanation"""
        # Create Occlusion explainer
        occlusion = Occlusion(self.model)
        
        # Get target class
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = output.argmax(dim=1).item()
        
        # Generate attribution
        attribution = occlusion.attribute(
            image,
            target=target_class,
            strides=(3, 3),
            sliding_window_shapes=(1, 15, 15)
        )
        
        # Convert to numpy and normalize
        attribution = attribution.squeeze(0).permute(1, 2, 0).cpu().numpy()
        attribution = np.abs(attribution).sum(axis=2)  # Sum across channels
        
        # Normalize
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        return attribution
    
    def generate_explanations(self, num_samples: int = 10) -> Dict[str, List[np.ndarray]]:
        """Generate explanations for multiple samples"""
        explanations = {method: [] for method in self.xai_methods.keys()}
        
        self.logger.info(f"Generating explanations for {num_samples} samples...")
        
        for i, (image, target) in enumerate(self.test_loader):
            if i >= num_samples:
                break
            
            image = image.to(self.device)
            target = target.item()
            
            self.logger.info(f"Processing sample {i+1}/{num_samples}, class: {target}")
            
            # Generate explanations for each method
            for method_name, method_func in self.xai_methods.items():
                try:
                    explanation = method_func(image, target)
                    explanations[method_name].append(explanation)
                except Exception as e:
                    self.logger.error(f"Error generating {method_name} explanation: {e}")
                    explanations[method_name].append(None)
        
        return explanations
    
    def visualize_explanations(self, explanations: Dict[str, List[np.ndarray]], 
                             save_dir: Path = None):
        """Visualize explanations"""
        if save_dir is None:
            save_dir = XAI_CONFIG['explanation_dir']
        
        save_dir.mkdir(exist_ok=True)
        
        # Get sample images
        sample_images = []
        for i, (image, target) in enumerate(self.test_loader):
            if i >= len(explanations[list(explanations.keys())[0]]):
                break
            sample_images.append((image.squeeze(0).permute(1, 2, 0).cpu().numpy(), target.item()))
        
        # Create visualizations for each method
        for method_name, method_explanations in explanations.items():
            self.logger.info(f"Visualizing {method_name} explanations...")
            
            # Create subplot for each sample
            num_samples = len(method_explanations)
            fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
            
            if num_samples == 1:
                axes = axes.reshape(2, 1)
            
            for i in range(num_samples):
                # Original image
                axes[0, i].imshow(sample_images[i][0])
                axes[0, i].set_title(f'Original (Class: {sample_images[i][1]})')
                axes[0, i].axis('off')
                
                # Explanation
                if method_explanations[i] is not None:
                    axes[1, i].imshow(method_explanations[i], cmap='jet')
                    axes[1, i].set_title(f'{method_name.upper()} Explanation')
                    axes[1, i].axis('off')
                else:
                    axes[1, i].text(0.5, 0.5, 'Error', ha='center', va='center')
                    axes[1, i].set_title(f'{method_name.upper()} Explanation')
                    axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_dir / f'{method_name}_explanations.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def compare_methods(self, explanations: Dict[str, List[np.ndarray]], 
                       save_dir: Path = None):
        """Compare different XAI methods"""
        if save_dir is None:
            save_dir = XAI_CONFIG['explanation_dir']
        
        save_dir.mkdir(exist_ok=True)
        
        # Get sample images
        sample_images = []
        for i, (image, target) in enumerate(self.test_loader):
            if i >= len(explanations[list(explanations.keys())[0]]):
                break
            sample_images.append((image.squeeze(0).permute(1, 2, 0).cpu().numpy(), target.item()))
        
        # Create comparison visualization
        num_samples = len(sample_images)
        num_methods = len(explanations)
        
        fig, axes = plt.subplots(num_samples, num_methods + 1, 
                                figsize=(4*(num_methods + 1), 4*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Original image
            axes[i, 0].imshow(sample_images[i][0])
            axes[i, 0].set_title(f'Original\n(Class: {sample_images[i][1]})')
            axes[i, 0].axis('off')
            
            # Explanations
            for j, (method_name, method_explanations) in enumerate(explanations.items()):
                if method_explanations[i] is not None:
                    axes[i, j + 1].imshow(method_explanations[i], cmap='jet')
                    axes[i, j + 1].set_title(f'{method_name.upper()}')
                    axes[i, j + 1].axis('off')
                else:
                    axes[i, j + 1].text(0.5, 0.5, 'Error', ha='center', va='center')
                    axes[i, j + 1].set_title(f'{method_name.upper()}')
                    axes[i, j + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_explanations(self, explanations: Dict[str, List[np.ndarray]], 
                         save_dir: Path = None):
        """Save explanations as numpy arrays"""
        if save_dir is None:
            save_dir = XAI_CONFIG['explanation_dir']
        
        save_dir.mkdir(exist_ok=True)
        
        for method_name, method_explanations in explanations.items():
            method_dir = save_dir / method_name
            method_dir.mkdir(exist_ok=True)
            
            for i, explanation in enumerate(method_explanations):
                if explanation is not None:
                    np.save(method_dir / f'explanation_{i}.npy', explanation)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate explanations for pneumonia detection model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--method', type=str, default='all',
                       choices=['all'] + XAI_CONFIG['methods'],
                       help='XAI method to use')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to explain')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for explanations')
    
    args = parser.parse_args()
    
    # Update config
    config = {
        'model_name': 'resnet50',  # Will be updated from checkpoint
        'xai': XAI_CONFIG
    }
    
    if args.output_dir:
        config['xai']['explanation_dir'] = Path(args.output_dir)
    
    # Filter methods if specific method requested
    if args.method != 'all':
        config['xai']['methods'] = [args.method]
    
    # Initialize explainer
    explainer = PneumoniaExplainer(args.model_path, config)
    
    # Generate explanations
    explanations = explainer.generate_explanations(args.num_samples)
    
    # Visualize explanations
    explainer.visualize_explanations(explanations)
    
    # Compare methods
    if len(explanations) > 1:
        explainer.compare_methods(explanations)
    
    # Save explanations
    if XAI_CONFIG['save_explanations']:
        explainer.save_explanations(explanations)
    
    print(f"Explanations generated and saved to {config['xai']['explanation_dir']}")

if __name__ == '__main__':
    main()