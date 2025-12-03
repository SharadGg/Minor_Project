"""Logging Utilities"""
import logging
from pathlib import Path

def setup_logger(name: str, log_dir: str = 'results/logs'):
    """Setup colored logger."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(Path(log_dir) / f'{name}.log')
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger