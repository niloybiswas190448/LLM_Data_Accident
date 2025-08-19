"""
Logging utilities for pneumonia detection
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import os
from datetime import datetime

def setup_logging(log_file: Optional[str] = None, 
                 level: str = 'INFO',
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
        format_string: Custom format string
        console_output: Whether to output to console
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('pneumonia_detection')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = 'pneumonia_detection') -> logging.Logger:
    """
    Get logger instance
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class TrainingLogger:
    """Logger specifically for training progress"""
    
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this training session
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'training_{timestamp}.log'
        
        # Setup logger
        self.logger = setup_logging(str(self.log_file))
        
        # Training metrics
        self.metrics = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'learning_rates': []
        }
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_acc: float, val_acc: float, lr: float = None):
        """Log epoch information"""
        self.logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                        f'Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                        f'Val Acc: {val_acc:.2f}%')
        
        # Store metrics
        self.metrics['train_losses'].append(train_loss)
        self.metrics['val_losses'].append(val_loss)
        self.metrics['train_accuracies'].append(train_acc)
        self.metrics['val_accuracies'].append(val_acc)
        if lr is not None:
            self.metrics['learning_rates'].append(lr)
    
    def log_batch(self, batch_idx: int, total_batches: int, loss: float, acc: float):
        """Log batch information"""
        if batch_idx % 10 == 0:  # Log every 10 batches
            self.logger.info(f'Batch {batch_idx}/{total_batches}: '
                           f'Loss: {loss:.4f}, Acc: {acc:.2f}%')
    
    def log_hyperparameters(self, config: dict):
        """Log hyperparameters"""
        self.logger.info("Training Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_model_info(self, model_name: str, num_params: int):
        """Log model information"""
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Number of parameters: {num_params:,}")
    
    def log_evaluation(self, metrics: dict):
        """Log evaluation results"""
        self.logger.info("Evaluation Results:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
    
    def save_metrics(self, save_path: str = None):
        """Save metrics to file"""
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.log_dir / f'metrics_{timestamp}.json'
        
        import json
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {save_path}")

class ExperimentLogger:
    """Logger for experiment tracking"""
    
    def __init__(self, experiment_name: str, log_dir: str = 'logs'):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'{experiment_name}_{timestamp}.log'
        
        # Setup logger
        self.logger = setup_logging(str(self.log_file))
        
        # Experiment metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': {},
            'results': {}
        }
    
    def log_config(self, config: dict):
        """Log experiment configuration"""
        self.metadata['config'] = config
        self.logger.info("Experiment Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_result(self, result_name: str, value: float):
        """Log experiment result"""
        self.metadata['results'][result_name] = value
        self.logger.info(f"Result - {result_name}: {value:.4f}")
    
    def log_model_performance(self, model_name: str, metrics: dict):
        """Log model performance"""
        self.logger.info(f"Model Performance - {model_name}:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
            self.metadata['results'][f'{model_name}_{metric}'] = value
    
    def save_experiment_summary(self):
        """Save experiment summary"""
        import json
        
        # Add end time
        self.metadata['end_time'] = datetime.now().isoformat()
        
        # Save summary
        summary_file = self.log_dir / 'experiment_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        self.logger.info(f"Experiment summary saved to {summary_file}")

def log_memory_usage(logger: logging.Logger):
    """Log memory usage"""
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def log_gpu_usage(logger: logging.Logger):
    """Log GPU usage if available"""
    try:
        import torch
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                memory_cached = torch.cuda.memory_reserved(i) / 1024 / 1024
                logger.info(f"GPU {i}: Allocated: {memory_allocated:.2f} MB, "
                           f"Cached: {memory_cached:.2f} MB")
        else:
            logger.info("No GPU available")
    except ImportError:
        logger.info("PyTorch not available for GPU monitoring")

def log_system_info(logger: logging.Logger):
    """Log system information"""
    import platform
    
    logger.info(f"System: {platform.system()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    
    # Log CPU info
    import psutil
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"CPU usage: {psutil.cpu_percent()}%")
    
    # Log memory info
    memory = psutil.virtual_memory()
    logger.info(f"Total memory: {memory.total / 1024 / 1024 / 1024:.2f} GB")
    logger.info(f"Available memory: {memory.available / 1024 / 1024 / 1024:.2f} GB")
    logger.info(f"Memory usage: {memory.percent}%")

def create_progress_logger(total_steps: int, description: str = "Progress"):
    """Create a progress logger with tqdm-like functionality"""
    from tqdm import tqdm
    
    class ProgressLogger:
        def __init__(self, total_steps: int, description: str):
            self.pbar = tqdm(total=total_steps, desc=description)
            self.logger = get_logger()
        
        def update(self, step: int = 1, **kwargs):
            """Update progress"""
            self.pbar.update(step)
            
            # Log additional information
            if kwargs:
                info_str = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
                self.logger.info(info_str)
        
        def close(self):
            """Close progress bar"""
            self.pbar.close()
    
    return ProgressLogger(total_steps, description)

def log_exception(logger: logging.Logger, exception: Exception, context: str = ""):
    """Log exception with context"""
    logger.error(f"Exception in {context}: {str(exception)}")
    logger.error(f"Exception type: {type(exception).__name__}")
    
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")

def log_performance_metrics(logger: logging.Logger, metrics: dict, prefix: str = ""):
    """Log performance metrics in a structured way"""
    logger.info(f"{prefix} Performance Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")

def log_data_info(logger: logging.Logger, dataset_info: dict):
    """Log dataset information"""
    logger.info("Dataset Information:")
    for key, value in dataset_info.items():
        logger.info(f"  {key}: {value}")

def log_model_architecture(logger: logging.Logger, model):
    """Log model architecture"""
    logger.info("Model Architecture:")
    logger.info(str(model))
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

def log_training_start(logger: logging.Logger, config: dict):
    """Log training start information"""
    logger.info("=" * 50)
    logger.info("TRAINING STARTED")
    logger.info("=" * 50)
    
    log_system_info(logger)
    log_config(logger, config)

def log_training_end(logger: logging.Logger, final_metrics: dict):
    """Log training end information"""
    logger.info("=" * 50)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 50)
    
    log_performance_metrics(logger, final_metrics, "Final")

def log_config(logger: logging.Logger, config: dict):
    """Log configuration"""
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

# Convenience function for quick logging setup
def quick_logger(name: str = 'pneumonia_detection', level: str = 'INFO'):
    """Quick setup for basic logging"""
    return setup_logging(level=level)