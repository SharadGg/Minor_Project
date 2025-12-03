"""Checkpoint Utilities"""
import torch
import os
from pathlib import Path

def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str):
    """Save model checkpoint."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    epoch = state['epoch']
    filepath = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_filepath)
        print(f"Best model saved: {best_filepath}")

def load_checkpoint(filepath: str) -> dict:
    """Load checkpoint."""
    return torch.load(filepath, map_location='cpu')