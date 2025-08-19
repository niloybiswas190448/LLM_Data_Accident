# Comparative Analysis of Deep Learning Models for Pneumonia Detection using Explainable AI

A comprehensive study comparing state-of-the-art deep learning models for chest X-ray pneumonia detection with explainable AI techniques to provide interpretable and trustworthy medical AI systems.

## 🎯 Project Overview

This project implements a comparative analysis framework for pneumonia detection using:
- **Multiple Deep Learning Models**: CNN, ResNet, DenseNet, EfficientNet, Vision Transformer
- **Public Datasets**: Chest X-ray datasets with pneumonia annotations
- **Explainable AI Techniques**: Grad-CAM, SHAP, LIME, Integrated Gradients
- **Comprehensive Evaluation**: Accuracy, sensitivity, specificity, AUC-ROC, interpretability metrics
- **Clinical Validation**: Radiologist-friendly visualizations and explanations

## 🚀 Key Features

### 🔬 **Multi-Model Comparison**
- CNN architectures (VGG, ResNet, DenseNet, EfficientNet)
- Vision Transformers (ViT, DeiT)
- Custom architectures optimized for medical imaging
- Ensemble methods for improved performance

### 🧠 **Explainable AI Integration**
- **Grad-CAM**: Class activation mapping for CNN interpretability
- **SHAP**: Shapley values for feature importance
- **LIME**: Local interpretable model-agnostic explanations
- **Integrated Gradients**: Attribution methods for deep networks
- **Attention Visualization**: Transformer attention maps

### 📊 **Comprehensive Evaluation**
- Standard metrics (Accuracy, Precision, Recall, F1-Score)
- Medical metrics (Sensitivity, Specificity, AUC-ROC)
- Interpretability metrics (Faithfulness, Monotonicity)
- Statistical significance testing
- Cross-validation and robustness analysis

### 🏥 **Clinical Relevance**
- Radiologist-friendly visualizations
- Region-of-interest highlighting
- Confidence scoring with uncertainty quantification
- Clinical decision support integration

## 📋 Datasets

### Primary Dataset: Chest X-Ray Images (Pneumonia)
- **Source**: [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size**: ~5,856 images
- **Classes**: Normal, Bacterial Pneumonia, Viral Pneumonia
- **Format**: JPEG images (various sizes)
- **License**: CC BY 4.0

### Additional Datasets:
1. **NIH Chest X-ray14**: 112,120 images, 14 disease classes
2. **CheXpert**: 224,316 images, 14 observations
3. **MIMIC-CXR**: 377,110 images, 14 labels

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Datasets      │───▶│   Data Pipeline │───▶│  Model Training │
│                 │    │                 │    │                 │
│ • Chest X-Ray   │    │ • Preprocessing │    │ • CNN Models    │
│ • NIH CXR14     │    │ • Augmentation  │    │ • ViT Models    │
│ • CheXpert      │    │ • Normalization │    │ • Custom Models │
│ • MIMIC-CXR     │    │ • Splitting     │    │ • Ensembles     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Evaluation    │    │   Explainable   │
                       │   Framework     │    │   AI Pipeline   │
                       │                 │    │                 │
                       │ • Metrics       │    │ • Grad-CAM      │
                       │ • Cross-val     │    │ • SHAP          │
                       │ • Statistical   │    │ • LIME          │
                       │ • Robustness    │    │ • Attention     │
                       └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Visualization │    │   Clinical      │
                       │   & Reporting   │    │   Integration   │
                       │                 │    │                 │
                       │ • Results Plots │    │ • Radiologist   │
                       │ • Comparison    │    │   Interface     │
                       │ • Heatmaps      │    │ • Decision      │
                       │ • Reports       │    │   Support       │
                       └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM
- 50GB+ disk space for datasets and models

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd pneumonia-detection-xai
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download datasets:**
```bash
python scripts/download_datasets.py
```

## 🚀 Quick Start

### 1. Data Preparation
```bash
python scripts/prepare_data.py --dataset chest_xray --output_dir data/
```

### 2. Model Training
```bash
python train.py --model resnet50 --dataset chest_xray --epochs 50
```

### 3. Explainable AI Analysis
```bash
python explain.py --model_path models/resnet50_best.pth --method gradcam
```

### 4. Comparative Analysis
```bash
python compare_models.py --models resnet50 densenet121 efficientnet_b0 vit_base
```

## 📊 Methodology

### Phase 1: Data Preparation
1. **Dataset Collection**: Download and organize chest X-ray datasets
2. **Preprocessing**: 
   - Image resizing and normalization
   - Data augmentation (rotation, scaling, brightness)
   - Train/validation/test splitting
3. **Quality Control**: Remove corrupted images, verify annotations

### Phase 2: Model Development
1. **Baseline Models**:
   - ResNet-50, ResNet-101
   - DenseNet-121, DenseNet-169
   - EfficientNet-B0, EfficientNet-B4
   - Vision Transformer (ViT-Base)

2. **Custom Architectures**:
   - Medical-specific attention mechanisms
   - Multi-scale feature fusion
   - Uncertainty quantification layers

3. **Training Strategy**:
   - Transfer learning from ImageNet
   - Learning rate scheduling
   - Early stopping and model checkpointing
   - Cross-validation

### Phase 3: Explainable AI Implementation
1. **Grad-CAM**: Generate class activation maps
2. **SHAP**: Calculate feature importance scores
3. **LIME**: Local interpretable explanations
4. **Integrated Gradients**: Attribution analysis
5. **Attention Visualization**: Transformer attention maps

### Phase 4: Evaluation Framework
1. **Performance Metrics**:
   - Accuracy, Precision, Recall, F1-Score
   - Sensitivity, Specificity, AUC-ROC
   - Cohen's Kappa, Matthews Correlation

2. **Interpretability Metrics**:
   - Faithfulness: How well explanations match model behavior
   - Monotonicity: Consistency of explanations
   - Sparsity: Conciseness of explanations

3. **Statistical Analysis**:
   - McNemar's test for model comparison
   - Confidence intervals
   - Effect size calculations

### Phase 5: Clinical Validation
1. **Radiologist Evaluation**: Expert review of explanations
2. **Region-of-Interest Analysis**: Focus on clinically relevant areas
3. **Confidence Scoring**: Uncertainty quantification
4. **Decision Support**: Integration with clinical workflow

## 📈 Expected Results

### Performance Comparison
| Model | Accuracy | Sensitivity | Specificity | AUC-ROC | F1-Score |
|-------|----------|-------------|-------------|---------|----------|
| ResNet-50 | 0.89 | 0.91 | 0.87 | 0.94 | 0.88 |
| DenseNet-121 | 0.91 | 0.93 | 0.89 | 0.95 | 0.90 |
| EfficientNet-B4 | 0.93 | 0.94 | 0.92 | 0.96 | 0.92 |
| ViT-Base | 0.92 | 0.93 | 0.91 | 0.95 | 0.91 |

### Interpretability Analysis
- **Grad-CAM**: Highlights lung regions and infiltrates
- **SHAP**: Identifies key radiographic features
- **LIME**: Provides local explanations for individual cases
- **Attention Maps**: Shows model focus areas

## 🔧 Configuration

### Model Configuration
```python
MODEL_CONFIG = {
    "resnet50": {
        "pretrained": True,
        "num_classes": 3,
        "dropout": 0.5,
        "learning_rate": 1e-4
    },
    "vit_base": {
        "patch_size": 16,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12
    }
}
```

### Training Configuration
```python
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "early_stopping": True
}
```

### XAI Configuration
```python
XAI_CONFIG = {
    "methods": ["gradcam", "shap", "lime", "integrated_gradients"],
    "target_layer": "layer4",
    "num_samples": 1000,
    "visualization": True
}
```

## 📊 Output Structure

```
outputs/
├── models/
│   ├── resnet50_best.pth
│   ├── densenet121_best.pth
│   └── vit_base_best.pth
├── results/
│   ├── performance_comparison.csv
│   ├── confusion_matrices/
│   └── roc_curves/
├── explanations/
│   ├── gradcam/
│   ├── shap/
│   ├── lime/
│   └── attention_maps/
├── visualizations/
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── explanation_heatmaps.png
└── reports/
    ├── comprehensive_analysis.pdf
    └── clinical_validation.pdf
```

## 🔍 Key Research Questions

1. **Performance**: Which deep learning architecture achieves the best pneumonia detection performance?
2. **Interpretability**: How do different XAI methods compare in providing clinically meaningful explanations?
3. **Robustness**: How robust are the models across different datasets and imaging conditions?
4. **Clinical Utility**: Do the explanations improve radiologist confidence and diagnostic accuracy?
5. **Generalization**: How well do models generalize to unseen data and different populations?

## 🏥 Clinical Impact

- **Improved Diagnosis**: Faster and more accurate pneumonia detection
- **Reduced Workload**: Automated screening of normal cases
- **Better Explanations**: Transparent AI decisions for clinical trust
- **Educational Tool**: Teaching aid for radiology trainees
- **Quality Assurance**: Second opinion for radiologists

## 🔒 Ethical Considerations

- **Data Privacy**: Anonymized medical images only
- **Bias Mitigation**: Diverse dataset representation
- **Clinical Validation**: Expert review of AI explanations
- **Transparency**: Open-source implementation and methodology
- **Responsible AI**: Uncertainty quantification and confidence scoring

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset providers and contributors
- Open-source deep learning frameworks
- Medical imaging research community
- Radiologists for clinical validation

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the research team
- Check the documentation and examples

---

**Note**: This system is designed for research and educational purposes. Clinical deployment requires additional validation and regulatory approval.