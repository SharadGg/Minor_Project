"""
Evaluation Script for Trained Models
CPU OPTIMIZED for Intel i3, 8GB RAM

Location: S:\Everything\Clg_Project\Final_Draft\PoseGestureAnalyzer\eval.py
"""

import torch
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import os
import sys

from src.models.model_factory import create_model
from datasets.custom_dataset import get_dataloaders
from src.training.metrics import calculate_metrics, plot_confusion_matrix
from src.utils.checkpoint import load_checkpoint


def evaluate(config_path: str, checkpoint_path: str):
    """Evaluate trained model on test set."""
    
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION")
    print("="*60 + "\n")
    
    # Load config
    print("🔧 Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cpu')
    print(f"🖥️  Using device: CPU\n")
    
    # Load model
    print("🧠 Loading model...")
    model = create_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        num_joints=config['model']['num_joints'],
        **config['model']['params']
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    
    # Handle DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"✓ Training best val accuracy: {checkpoint['best_val_acc']:.4f}\n")
    
    # Load test data
    print("📦 Loading test data...")
    _, _, test_loader = get_dataloaders(config)
    print(f"✓ Test samples: {len(test_loader.dataset)}\n")
    
    # Evaluate
    print("🔍 Evaluating on test set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Testing"):
            data = data.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds)
    
    # Display results
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS")
    print("="*60)
    print(f"\n{'Metric':<20} {'Value':<10}")
    print("-" * 35)
    print(f"{'Accuracy':<20} {metrics['accuracy']:.4f}")
    print(f"{'Precision (weighted)':<20} {metrics['precision']:.4f}")
    print(f"{'Recall (weighted)':<20} {metrics['recall']:.4f}")
    print(f"{'F1-Score (weighted)':<20} {metrics['f1']:.4f}")
    print("\n" + "="*60)
    
    # Interpretation
    accuracy = metrics['accuracy']
    print("\n💡 Interpretation:")
    if accuracy >= 0.75:
        print("   ✅ Excellent! Model performs very well.")
    elif accuracy >= 0.65:
        print("   ✓ Good! Acceptable performance for synthetic data.")
    elif accuracy >= 0.50:
        print("   ⚠️  Fair. Model learned patterns but has room for improvement.")
    else:
        print("   ❌ Poor. Model may need more training or better data.")
    
    print(f"\n   Note: With synthetic data, 65-75% accuracy is expected and acceptable.")
    print(f"   Real-world data would typically achieve 80-90% with this architecture.")
    
    # Plot confusion matrix
    print("\n📊 Generating confusion matrix...")
    os.makedirs('results/plots', exist_ok=True)
    cm_path = 'results/plots/confusion_matrix.png'
    plot_confusion_matrix(
        all_labels,
        all_preds,
        save_path=cm_path
    )
    print(f"✓ Confusion matrix saved: {cm_path}")
    
    # Per-class analysis
    print("\n📋 Per-Class Performance:")
    print("-" * 60)
    
    # Get class labels
    emotion_labels = ['Happy', 'Sad', 'Angry', 'Confused', 'Neutral', 'Excited', 'Fearful']
    gesture_labels = ['Waving', 'Pointing', 'Asking', 'Signaling', 'Warning', 
                     'Greeting', 'Dismissing', 'Celebrating']
    all_labels_list = emotion_labels + gesture_labels
    
    # Calculate per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 65)
    
    for i, label in enumerate(all_labels_list):
        if i < len(precision):
            print(f"{label:<15} {precision[i]:<12.3f} {recall[i]:<12.3f} {f1[i]:<12.3f} {support[i]:<10}")
    
    print("\n" + "="*60 + "\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Trained Model')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Run evaluation
    evaluate(args.config, args.checkpoint)
    
    print("✅ Evaluation complete!\n")


if __name__ == '__main__':
    main()