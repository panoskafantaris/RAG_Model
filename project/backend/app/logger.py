"""
Centralized logging configuration.
"""
import logging
import sys
from .config import LOG_LEVEL, LOG_FORMAT

def setup_logger(name: str) -> logging.Logger:
    """
    Create a configured logger.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, LOG_LEVEL))
    
    # Formatter
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger