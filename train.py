"""
Complete Training Pipeline for Pose-based Gesture Recognition
CPU OPTIMIZED for Intel i3, 8GB RAM, No GPU

Location: S:\Everything\Clg_Project\Final_Draft\PoseGestureAnalyzer\train.py

Features:
- CPU-only training (no CUDA)
- Small batch size (4) for 8GB RAM
- Gradient accumulation for effective batch size
- Early stopping
- TensorBoard logging
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import yaml
import os
from tqdm import tqdm
from pathlib import Path
import warnings
import time
warnings.filterwarnings('ignore')

# Import custom modules
from src.models.model_factory import create_model
from datasets.custom_dataset import GestureDataset, get_dataloaders
from src.training.losses import FocalLoss, LabelSmoothingCrossEntropy
from src.training.metrics import calculate_metrics, plot_confusion_matrix
from src.utils.logger import setup_logger
from src.utils.checkpoint import save_checkpoint, load_checkpoint


class CPUTrainer:
    """
    CPU-Optimized Trainer class for systems without GPU.
    Designed for Intel i3-1115G4, 8GB RAM.
    """
    def __init__(self, config):
        self.config = config
        
        # Force CPU device
        self.device = torch.device('cpu')
        print(f"\n{'='*60}")
        print(f"🖥️  RUNNING ON CPU (No GPU detected or configured)")
        print(f"{'='*60}\n")
        
        # Setup logging
        self.logger = setup_logger('Trainer', config['logging']['log_dir'])
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Training optimized for CPU")
        
        # Initialize TensorBoard
        self.writer = SummaryWriter(log_dir=config['logging']['tensorboard_dir'])
        
        # Build model
        self.model = self._build_model()
        self.model = self.model.to(self.device)
        
        # Setup loss function
        self.criterion = self._build_criterion()
        
        # Setup optimizer
        self.optimizer = self._build_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._build_scheduler()
        
        # Gradient accumulation (important for small batch size)
        self.accumulation_steps = config['training']['gradient_accumulation_steps']
        self.logger.info(f"Gradient accumulation steps: {self.accumulation_steps}")
        self.logger.info(f"Effective batch size: {config['training']['batch_size'] * self.accumulation_steps}")
        
        # Early stopping
        self.early_stopping_patience = config['training']['early_stopping_patience']
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Tracking
        self.epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        
        # Training start time
        self.start_time = None
        
    def _build_model(self):
        """Build model from config."""
        self.logger.info("Building model...")
        model = create_model(
            model_name=self.config['model']['name'],
            num_classes=self.config['model']['num_classes'],
            num_joints=self.config['model']['num_joints'],
            **self.config['model']['params']
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model: {self.config['model']['name']}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Model size: {total_params * 4 / (1024**2):.2f} MB")
        
        return model
    
    def _build_criterion(self):
        """Build loss function."""
        loss_type = self.config['training']['loss']
        
        if loss_type == 'crossentropy':
            criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config['training'].get('label_smoothing', 0.0)
            )
        elif loss_type == 'focal':
            criterion = FocalLoss(
                alpha=self.config['training'].get('focal_alpha', 0.25),
                gamma=self.config['training'].get('focal_gamma', 2.0)
            )
        elif loss_type == 'label_smoothing':
            criterion = LabelSmoothingCrossEntropy(
                smoothing=self.config['training']['label_smoothing']
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.logger.info(f"Loss function: {loss_type}")
        return criterion
    
    def _build_optimizer(self):
        """Build optimizer."""
        opt_type = self.config['optimizer']['type']
        lr = self.config['optimizer']['lr']
        weight_decay = self.config['optimizer']['weight_decay']
        
        if opt_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif opt_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
        
        self.logger.info(f"Optimizer: {opt_type} (lr={lr}, wd={weight_decay})")
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        sched_type = self.config['scheduler']['type']
        
        if sched_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config['scheduler']['T_0'],
                T_mult=self.config['scheduler']['T_mult']
            )
        elif sched_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['scheduler']['step_size'],
                gamma=self.config['scheduler']['gamma']
            )
        elif sched_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None
        
        self.logger.info(f"Scheduler: {sched_type}")
        return scheduler
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        epoch_start = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}/{self.config['training']['epochs']}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Track metrics
            total_loss += loss.item() * self.accumulation_steps
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.accumulation_steps:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to TensorBoard
            if self.global_step % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('train/batch_loss', loss.item() * self.accumulation_steps, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        metrics = calculate_metrics(all_labels, all_preds)
        
        epoch_time = time.time() - epoch_start
        self.logger.info(f"Epoch {self.epoch} completed in {epoch_time/60:.2f} minutes")
        
        return avg_loss, metrics
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Validation"):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metrics = calculate_metrics(all_labels, all_preds)
        
        return avg_loss, metrics, all_labels, all_preds
    
    def train(self, train_loader, val_loader):
        """Complete training loop."""
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.config['training']['epochs']}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        self.start_time = time.time()
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            self.epoch = epoch
            
            print(f"\n{'='*60}")
            print(f"📊 Epoch {epoch}/{self.config['training']['epochs']}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics, val_labels, val_preds = self.validate(val_loader)
            
            # Log metrics
            log_msg = (
                f"Epoch {self.epoch}: "
                f"Train Loss={train_loss:.4f}, Train Acc={train_metrics['accuracy']:.4f} | "
                f"Val Loss={val_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}"
            )
            self.logger.info(log_msg)
            print(f"\n📈 {log_msg}\n")
            
            # TensorBoard logging
            self.writer.add_scalar('epoch/train_loss', train_loss, self.epoch)
            self.writer.add_scalar('epoch/train_acc', train_metrics['accuracy'], self.epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, self.epoch)
            self.writer.add_scalar('epoch/val_acc', val_metrics['accuracy'], self.epoch)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                print(f"✨ New best model! Validation accuracy: {self.best_val_acc:.4f}")
            
            save_checkpoint(
                {
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_acc': self.best_val_acc,
                    'config': self.config
                },
                is_best=is_best,
                checkpoint_dir=self.config['logging']['checkpoint_dir']
            )
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                self.logger.info(f"Early stopping counter: {self.patience_counter}/{self.early_stopping_patience}")
                
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {self.epoch} epochs")
                print(f"\n⏹️  Early stopping triggered after {self.epoch} epochs")
                break
            
            # Save confusion matrix periodically
            if self.epoch % self.config['logging']['save_freq'] == 0:
                plot_confusion_matrix(
                    val_labels,
                    val_preds,
                    save_path=os.path.join(
                        self.config['logging']['plot_dir'],
                        f'confusion_matrix_epoch_{self.epoch}.png'
                    )
                )
        
        # Training complete
        total_time = time.time() - self.start_time
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED!")
        print("="*60)
        self.logger.info("Training completed!")
        self.logger.info(f"Total training time: {total_time/3600:.2f} hours")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"\n⏱️  Total training time: {total_time/3600:.2f} hours")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"💾 Best model saved at: {self.config['logging']['checkpoint_dir']}/best_model.pth")
        print("\n" + "="*60 + "\n")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Pose-based Gesture Recognition (CPU Optimized)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()
    
    # Load configuration
    print("\n🔧 Loading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['tensorboard_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['plot_dir'], exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    print("📦 Loading datasets...")
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    print(f"✓ Training samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    # Initialize trainer
    trainer = CPUTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.epoch = checkpoint['epoch']
        trainer.best_val_acc = checkpoint['best_val_acc']
        print(f"✓ Resumed from epoch {trainer.epoch}")
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()