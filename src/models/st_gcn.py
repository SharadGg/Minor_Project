"""
ST-GCN: Spatial Temporal Graph Convolutional Networks
Paper: https://arxiv.org/abs/1801.07455

Complete implementation for skeleton-based action recognition.
Author: Final Year Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer for skeleton data.
    Applies convolution on graph-structured data.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int]):
        super(GraphConvolution, self).__init__()
        
        self.kernel_size = kernel_size
        
        # Spatial kernel: partitions graph into K subsets
        # Temporal kernel: convolution across time dimension
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size[1],  # K spatial subsets
            kernel_size=(kernel_size[0], 1),  # Temporal only
            padding=(kernel_size[0] // 2, 0)
        )
        
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (N, C, T, V) where:
               N = batch size, C = channels, T = time steps, V = vertices (joints)
            A: Adjacency matrix (K, V, V) where K = number of spatial subsets
        
        Returns:
            Output features (N, C_out, T, V)
        """
        # Apply temporal convolution
        x = self.conv(x)
        
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size[1], kc // self.kernel_size[1], t, v)
        
        # Apply spatial graph convolution using adjacency matrix
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        
        return x.contiguous()


class STGCNBlock(nn.Module):
    """
    ST-GCN Block: Spatial-Temporal Graph Convolution Block
    
    Architecture:
    Input -> GCN -> BatchNorm -> ReLU -> Temporal Conv -> BatchNorm -> ReLU -> Dropout -> Output
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        dropout: float = 0.5,
        residual: bool = True
    ):
        super(STGCNBlock, self).__init__()
        
        self.gcn = GraphConvolution(in_channels, out_channels, kernel_size)
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (9, 1),  # Temporal kernel size
                (stride, 1),
                padding=(4, 0)
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )
        
        # Residual connection
        self.residual = residual
        if not residual:
            self.residual_conv = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual_conv = lambda x: x
        else:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ST-GCN block.
        
        Args:
            x: Input tensor (N, C, T, V)
            A: Adjacency matrix (K, V, V)
        
        Returns:
            Output tensor (N, C_out, T, V)
        """
        res = self.residual_conv(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)


class STGCN(nn.Module):
    """
    Complete ST-GCN Model for Pose-based Action Recognition
    
    Architecture:
        Input (N, C, T, V) -> 
        Multiple ST-GCN Blocks ->
        Global Average Pooling ->
        Fully Connected Layers ->
        Output (N, num_classes)
    """
    def __init__(
        self,
        num_class: int,
        num_point: int = 25,  # Number of joints (25 for full body)
        num_person: int = 1,
        in_channels: int = 3,  # (x, y, confidence) or (x, y, z)
        graph_args: dict = None,
        edge_importance_weighting: bool = True,
        dropout: float = 0.5
    ):
        super(STGCN, self).__init__()
        
        if graph_args is None:
            graph_args = {'layout': 'mediapipe', 'strategy': 'spatial'}
        
        # Build graph structure
        from .graph import Graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        # Edge importance weighting (learnable)
        self.edge_importance_weighting = edge_importance_weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in range(10)  # 10 ST-GCN blocks
            ])
        else:
            self.edge_importance = [1] * 10
        
        # Network architecture
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        # Input batch normalization
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        # ST-GCN blocks with increasing channels
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, kernel_size, 1, dropout, residual=False),
            STGCNBlock(64, 64, kernel_size, 1, dropout),
            STGCNBlock(64, 64, kernel_size, 1, dropout),
            STGCNBlock(64, 64, kernel_size, 1, dropout),
            STGCNBlock(64, 128, kernel_size, 2, dropout),
            STGCNBlock(128, 128, kernel_size, 1, dropout),
            STGCNBlock(128, 128, kernel_size, 1, dropout),
            STGCNBlock(128, 256, kernel_size, 2, dropout),
            STGCNBlock(256, 256, kernel_size, 1, dropout),
            STGCNBlock(256, 256, kernel_size, 1, dropout),
        ))
        
        # Classification head
        self.fc = nn.Linear(256, num_class)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, V, M) where:
               N = batch size
               C = input channels (3 for x,y,conf or x,y,z)
               T = number of frames
               V = number of vertices/joints
               M = number of persons
        
        Returns:
            Output logits (N, num_classes)
        """
        # Data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)
        
        # Forward through ST-GCN blocks
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)
        
        # Global pooling
        x = F.avg_pool2d(x, x.size()[2:])  # (N*M, C, 1, 1)
        x = x.view(N, M, -1).mean(dim=1)   # Average across persons
        
        # Classification
        x = self.fc(x)
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representations before classification."""
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)
        
        x = F.avg_pool2d(x, x.size()[2:])
        features = x.view(N, M, -1).mean(dim=1)
        
        return features


def create_stgcn_model(
    num_classes: int,
    num_joints: int = 25,
    input_channels: int = 3,
    dropout: float = 0.5
) -> STGCN:
    """
    Factory function to create ST-GCN model.
    
    Args:
        num_classes: Number of output classes (emotions + gestures)
        num_joints: Number of skeleton joints
        input_channels: Number of input channels (3 for x,y,conf)
        dropout: Dropout rate
    
    Returns:
        Initialized ST-GCN model
    """
    model = STGCN(
        num_class=num_classes,
        num_point=num_joints,
        in_channels=input_channels,
        dropout=dropout,
        edge_importance_weighting=True
    )
    return model


if __name__ == "__main__":
    # Test the model
    model = create_stgcn_model(num_classes=15, num_joints=25)
    
    # Create dummy input: (batch=4, channels=3, frames=300, joints=25, persons=1)
    dummy_input = torch.randn(4, 3, 300, 25, 1)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")