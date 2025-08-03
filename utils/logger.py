"""
Logging utility for Road Accident Analysis Pipeline
Provides consistent logging configuration and utilities
"""

import logging
import os
from datetime import datetime
from config import LOGGING_CONFIG

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Set up a logger with consistent configuration
    
    Args:
        name: Logger name
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers if they already exist
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, LOGGING_CONFIG["level"]))
    
    # Create formatter
    formatter = logging.Formatter(LOGGING_CONFIG["format"])
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_pipeline_logger() -> logging.Logger:
    """
    Get the main pipeline logger
    
    Returns:
        Configured logger for the main pipeline
    """
    return setup_logger("road_accident_pipeline", LOGGING_CONFIG["file"])

def log_execution_time(func):
    """
    Decorator to log function execution time
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_pipeline_logger()
        start_time = datetime.now()
        logger.info(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.error(f"Error in {func.__name__} after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper