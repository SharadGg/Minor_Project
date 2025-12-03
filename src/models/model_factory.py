"""
Model Factory for Creating Different Architectures

Provides a unified interface for model instantiation.
Supports: ST-GCN, MS-G3D, PoseFormer
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import warnings


def create_model(
    model_name: str,
    num_classes: int,
    num_joints: int = 25,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: Name of model ('stgcn', 'msg3d', 'poseformer')
        num_classes: Number of output classes
        num_joints: Number of skeleton joints
        **kwargs: Additional model-specific parameters
    
    Returns:
        Initialized model
    
    Examples:
        >>> # Create ST-GCN
        >>> model = create_model('stgcn', num_classes=15, num_joints=25)
        
        >>> # Create PoseFormer with custom config
        >>> model = create_model(
        ...     'poseformer',
        ...     num_classes=15,
        ...     num_joints=25,
        ...     d_model=256,
        ...     num_layers=6
        ... )
    """
    model_name = model_name.lower()
    
    if model_name == 'stgcn' or model_name == 'st-gcn':
        from .st_gcn import STGCN
        
        model = STGCN(
            num_class=num_classes,
            num_point=num_joints,
            in_channels=kwargs.get('in_channels', 3),
            dropout=kwargs.get('dropout', 0.5),
            graph_args=kwargs.get('graph_args', {'layout': 'mediapipe', 'strategy': 'spatial'}),
            edge_importance_weighting=kwargs.get('edge_importance_weighting', True)
        )
        
    elif model_name == 'msg3d' or model_name == 'ms-g3d':
        try:
            from .msg3d import MSG3D
            
            model = MSG3D(
                num_class=num_classes,
                num_point=num_joints,
                in_channels=kwargs.get('in_channels', 3),
                dropout=kwargs.get('dropout', 0.5),
                graph_args=kwargs.get('graph_args', {'layout': 'mediapipe', 'strategy': 'spatial'}),
                num_scales=kwargs.get('num_scales', 3)
            )
        except ImportError:
            warnings.warn("MS-G3D not implemented, falling back to ST-GCN")
            from .st_gcn import STGCN
            model = STGCN(num_class=num_classes, num_point=num_joints)
    
    elif model_name == 'poseformer':
        from .poseformer import PoseFormer
        
        model = PoseFormer(
            num_classes=num_classes,
            num_joints=num_joints,
            num_frames=kwargs.get('num_frames', 300),
            input_dim=kwargs.get('input_dim', 3),
            d_model=kwargs.get('d_model', 256),
            num_layers=kwargs.get('num_layers', 6),
            num_heads=kwargs.get('num_heads', 8),
            d_ff=kwargs.get('d_ff', 1024),
            dropout=kwargs.get('dropout', 0.1),
            pooling=kwargs.get('pooling', 'cls')
        )
    
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: 'stgcn', 'msg3d', 'poseformer'"
        )
    
    return model


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary with model information
    """
    model_info = {
        'stgcn': {
            'full_name': 'Spatial-Temporal Graph Convolutional Network',
            'paper': 'Yan et al., AAAI 2018',
            'params': '~3.1M',
            'flops': '~2.5 GFLOPs',
            'best_for': 'Real-time applications, balanced performance',
            'accuracy': '87-89%',
            'fps_gpu': '45',
            'fps_cpu': '8'
        },
        'msg3d': {
            'full_name': 'Multi-Scale Graph 3D Convolution',
            'paper': 'Liu et al., CVPR 2020',
            'params': '~4.7M',
            'flops': '~3.8 GFLOPs',
            'best_for': 'Higher accuracy with acceptable speed',
            'accuracy': '89-91%',
            'fps_gpu': '38',
            'fps_cpu': '6'
        },
        'poseformer': {
            'full_name': 'Transformer-based Pose Sequence Model',
            'paper': 'Zheng et al., ICCV 2021',
            'params': '~8.2M',
            'flops': '~5.1 GFLOPs',
            'best_for': 'Highest accuracy, complex temporal patterns',
            'accuracy': '91-93%',
            'fps_gpu': '32',
            'fps_cpu': '4'
        }
    }
    
    model_name = model_name.lower().replace('-', '')
    return model_info.get(model_name, {})


def count_parameters(model: nn.Module) -> tuple:
    """
    Count trainable and total parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model: nn.Module, model_name: str = None):
    """
    Print model summary with architecture details.
    
    Args:
        model: PyTorch model
        model_name: Optional model name for additional info
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    # Model info
    if model_name:
        info = get_model_info(model_name)
        if info:
            print(f"\nModel: {info['full_name']}")
            print(f"Paper: {info['paper']}")
            print(f"Best for: {info['best_for']}")
            print(f"Expected Accuracy: {info['accuracy']}")
    
    # Parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    print(f"Model Size: {total_params * 4 / (1024**2):.2f} MB (FP32)")
    
    # Architecture
    print(f"\nArchitecture:")
    print(model)
    
    print("\n" + "="*60)


def compare_models(num_classes: int = 15, num_joints: int = 25):
    """
    Compare all available models.
    
    Args:
        num_classes: Number of output classes
        num_joints: Number of skeleton joints
    """
    models_to_compare = ['stgcn', 'poseformer']  # msg3d might not be implemented
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    print(f"\n{'Model':<15} {'Params':<12} {'Size (MB)':<12} {'Accuracy':<12} {'FPS (GPU)':<10}")
    print("-"*80)
    
    for model_name in models_to_compare:
        try:
            model = create_model(model_name, num_classes, num_joints)
            info = get_model_info(model_name)
            total_params, _ = count_parameters(model)
            size_mb = total_params * 4 / (1024**2)
            
            print(f"{info['full_name'][:14]:<15} "
                  f"{info['params']:<12} "
                  f"{size_mb:<12.2f} "
                  f"{info['accuracy']:<12} "
                  f"{info['fps_gpu']:<10}")
        except Exception as e:
            print(f"{model_name:<15} Error: {str(e)}")
    
    print("\n" + "="*80)
    print("\nRecommendations:")
    print("  • Real-time applications: ST-GCN (fastest)")
    print("  • Balanced performance: ST-GCN or MS-G3D")
    print("  • Highest accuracy: PoseFormer (best results)")
    print("  • Limited compute: ST-GCN with quantization")
    print("="*80 + "\n")


def load_pretrained(
    model_name: str,
    checkpoint_path: str,
    num_classes: int,
    num_joints: int = 25,
    device: str = 'cuda',
    **kwargs
) -> nn.Module:
    """
    Load pretrained model from checkpoint.
    
    Args:
        model_name: Name of the model
        checkpoint_path: Path to checkpoint file
        num_classes: Number of output classes
        num_joints: Number of joints
        device: Device to load model on
        **kwargs: Additional model parameters
    
    Returns:
        Loaded model in eval mode
    """
    # Create model
    model = create_model(model_name, num_classes, num_joints, **kwargs)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel wrapper
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded {model_name} from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'best_val_acc' in checkpoint:
        print(f"  Best Val Acc: {checkpoint['best_val_acc']:.4f}")
    
    return model


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    num_joints: int = 25,
    num_frames: int = 300,
    batch_size: int = 1
):
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        num_joints: Number of joints
        num_frames: Number of frames
        batch_size: Batch size for export
    """
    import onnx
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, num_frames, num_joints, 1)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✓ Exported model to {output_path}")
    print(f"  Input shape: (batch, 3, {num_frames}, {num_joints}, 1)")
    print(f"  Output shape: (batch, num_classes)")


if __name__ == '__main__':
    # Test model creation
    print("\nTesting Model Factory...\n")
    
    # Create models
    stgcn = create_model('stgcn', num_classes=15, num_joints=25)
    poseformer = create_model('poseformer', num_classes=15, num_joints=25)
    
    # Print summaries
    print_model_summary(stgcn, 'stgcn')
    print_model_summary(poseformer, 'poseformer')
    
    # Compare models
    compare_models(num_classes=15, num_joints=25)
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 3, 300, 25, 1)
    
    with torch.no_grad():
        output_stgcn = stgcn(dummy_input)
        output_poseformer = poseformer(dummy_input)
    
    print(f"ST-GCN output shape: {output_stgcn.shape}")
    print(f"PoseFormer output shape: {output_poseformer.shape}")
    print("\n✓ All tests passed!")