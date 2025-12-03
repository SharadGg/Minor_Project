"""Custom Dataset Class"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import random

class GestureDataset(Dataset):
    """Dataset for gesture recognition."""
    def __init__(self, data_path: str, split: str = 'train', window_size: int = 150, augment: bool = False):
        self.data_path = Path(data_path) / split
        self.window_size = window_size
        self.augment = augment and (split == 'train')
        
        self.samples = []
        self.labels = []
        
        if self.data_path.exists():
            for file in self.data_path.glob('*.npy'):
                label = int(file.stem.split('_')[0])
                self.samples.append(str(file))
                self.labels.append(label)
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sequence = np.load(self.samples[idx])
        label = self.labels[idx]
        
        # Pad or crop to window_size
        if len(sequence) < self.window_size:
            padding = np.zeros((self.window_size - len(sequence), *sequence.shape[1:]))
            sequence = np.concatenate([sequence, padding], axis=0)
        elif len(sequence) > self.window_size:
            start = random.randint(0, len(sequence) - self.window_size) if self.augment else (len(sequence) - self.window_size) // 2
            sequence = sequence[start:start + self.window_size]
        
        if self.augment:
            sequence = self.augment_sequence(sequence)
        
        sequence = torch.FloatTensor(sequence).permute(2, 0, 1).unsqueeze(-1)
        return sequence, label
    
    def augment_sequence(self, sequence):
        """Apply data augmentation."""
        if random.random() < 0.5:
            angle = random.uniform(-15, 15) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            sequence[:, :, :2] = sequence[:, :, :2] @ rotation_matrix.T
        if random.random() < 0.5:
            sequence[:, :, :2] *= random.uniform(0.9, 1.1)
        return sequence

def get_dataloaders(config):
    """Create dataloaders."""
    train_dataset = GestureDataset(config['data']['data_dir'], 'train', config['data']['window_size'], config['data']['augmentation'])
    val_dataset = GestureDataset(config['data']['data_dir'], 'val', config['data']['window_size'], False)
    test_dataset = GestureDataset(config['data']['data_dir'], 'test', config['data']['window_size'], False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0, pin_memory=False)
    
    return train_loader, val_loader, test_loader