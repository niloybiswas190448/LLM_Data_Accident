"""
Model architectures for pneumonia detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Any, Optional
import timm

class CustomResNet(nn.Module):
    """Custom ResNet with modifications for medical imaging"""
    
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 3, 
                 pretrained: bool = True, dropout: float = 0.5):
        super(CustomResNet, self).__init__()
        
        # Load pretrained model
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Get feature dimension
        feature_dim = self.backbone.fc.in_features
        
        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class CustomDenseNet(nn.Module):
    """Custom DenseNet with modifications for medical imaging"""
    
    def __init__(self, model_name: str = 'densenet121', num_classes: int = 3,
                 pretrained: bool = True, dropout: float = 0.5):
        super(CustomDenseNet, self).__init__()
        
        # Load pretrained model
        if model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
        elif model_name == 'densenet169':
            self.backbone = models.densenet169(pretrained=pretrained)
        elif model_name == 'densenet201':
            self.backbone = models.densenet201(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported DenseNet model: {model_name}")
        
        # Get feature dimension
        feature_dim = self.backbone.classifier.in_features
        
        # Replace final layer
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class CustomEfficientNet(nn.Module):
    """Custom EfficientNet with modifications for medical imaging"""
    
    def __init__(self, model_name: str = 'efficientnet_b0', num_classes: int = 3,
                 pretrained: bool = True, dropout: float = 0.5):
        super(CustomEfficientNet, self).__init__()
        
        # Load pretrained model
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        return self.classifier(features)

class VisionTransformer(nn.Module):
    """Vision Transformer for medical imaging"""
    
    def __init__(self, model_name: str = 'vit_base_patch16_224', num_classes: int = 3,
                 pretrained: bool = True, dropout: float = 0.1):
        super(VisionTransformer, self).__init__()
        
        # Load pretrained model
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        # Use CLS token for classification
        cls_token = features[:, 0]
        return self.classifier(cls_token)

class MedicalAttentionNet(nn.Module):
    """Custom attention-based network for medical imaging"""
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.5):
        super(MedicalAttentionNet, self).__init__()
        
        # Backbone (ResNet-50 without final layer)
        backbone = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Classify
        output = self.classifier(attended_features)
        
        return output

class MultiScaleNet(nn.Module):
    """Multi-scale feature fusion network"""
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.5):
        super(MultiScaleNet, self).__init__()
        
        # Backbone (ResNet-50)
        backbone = models.resnet50(pretrained=True)
        self.layer1 = nn.Sequential(*list(backbone.children())[:5])   # 256 channels
        self.layer2 = nn.Sequential(*list(backbone.children())[5:6])  # 512 channels
        self.layer3 = nn.Sequential(*list(backbone.children())[6:7])  # 1024 channels
        self.layer4 = nn.Sequential(*list(backbone.children())[7:8])  # 2048 channels
        
        # Multi-scale feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(2048 + 1024 + 512, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract multi-scale features
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        
        # Upsample and concatenate
        f3_up = F.interpolate(f3, size=f4.shape[2:], mode='bilinear', align_corners=False)
        f2_up = F.interpolate(f2, size=f4.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate features
        fused = torch.cat([f4, f3_up, f2_up], dim=1)
        
        # Fusion and classification
        fused = self.fusion(fused)
        output = self.classifier(fused)
        
        return output

class UncertaintyNet(nn.Module):
    """Network with uncertainty quantification"""
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.5):
        super(UncertaintyNet, self).__init__()
        
        # Backbone
        backbone = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Mean and variance heads
        feature_dim = 2048
        self.mean_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        self.var_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
            nn.Softplus()  # Ensure positive variance
        )
    
    def forward(self, x):
        features = self.features(x)
        
        mean = self.mean_head(features)
        var = self.var_head(features)
        
        return mean, var

class EnsembleNet(nn.Module):
    """Ensemble of multiple models"""
    
    def __init__(self, model_configs: list, num_classes: int = 3):
        super(EnsembleNet, self).__init__()
        
        self.models = nn.ModuleList()
        for config in model_configs:
            model = get_model(config['name'], config)
            self.models.append(model)
        
        # Ensemble weights (learnable)
        self.weights = nn.Parameter(torch.ones(len(model_configs)) / len(model_configs))
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Weighted ensemble
        weighted_outputs = []
        for i, output in enumerate(outputs):
            weighted_outputs.append(self.weights[i] * output)
        
        ensemble_output = sum(weighted_outputs)
        return ensemble_output

def get_model(model_name: str, config: Dict[str, Any]) -> nn.Module:
    """
    Get model based on name and configuration
    
    Args:
        model_name: Name of the model architecture
        config: Model configuration dictionary
    
    Returns:
        PyTorch model
    """
    num_classes = config.get('num_classes', 3)
    pretrained = config.get('pretrained', True)
    dropout = config.get('dropout', 0.5)
    
    if model_name.startswith('resnet'):
        return CustomResNet(model_name, num_classes, pretrained, dropout)
    
    elif model_name.startswith('densenet'):
        return CustomDenseNet(model_name, num_classes, pretrained, dropout)
    
    elif model_name.startswith('efficientnet'):
        return CustomEfficientNet(model_name, num_classes, pretrained, dropout)
    
    elif model_name.startswith('vit'):
        return VisionTransformer(model_name, num_classes, pretrained, dropout)
    
    elif model_name == 'medical_attention':
        return MedicalAttentionNet(num_classes, dropout)
    
    elif model_name == 'multi_scale':
        return MultiScaleNet(num_classes, dropout)
    
    elif model_name == 'uncertainty':
        return UncertaintyNet(num_classes, dropout)
    
    elif model_name == 'ensemble':
        model_configs = config.get('model_configs', [])
        return EnsembleNet(model_configs, num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> str:
    """Get model summary"""
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = -1
            
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = -1
            
            params = 0
            for p in module.parameters():
                params += p.numel()
            
            summary[m_key]["params"] = params
        
        hooks.append(module.register_forward_hook(hook))
    
    # Create hooks
    summary = OrderedDict()
    hooks = []
    
    # Register hook
    model.apply(register_hook)
    
    # Make a forward pass
    x = torch.zeros(input_size)
    model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Create summary string
    summary_str = "Model Summary:\n"
    summary_str += "=" * 80 + "\n"
    summary_str += f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<15}\n"
    summary_str += "=" * 80 + "\n"
    
    total_params = 0
    trainable_params = 0
    
    for layer in summary:
        summary_str += f"{layer:<25} {str(summary[layer]['output_shape']):<25} {summary[layer]['params']:<15}\n"
        total_params += summary[layer]["params"]
    
    summary_str += "=" * 80 + "\n"
    summary_str += f"Total params: {total_params:,}\n"
    summary_str += f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
    summary_str += f"Non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}\n"
    
    return summary_str

# Import OrderedDict for model summary
from collections import OrderedDict