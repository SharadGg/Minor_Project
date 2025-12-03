"""
PoseFormer: 3D Human Pose Estimation with Transformers
Paper: https://arxiv.org/abs/2103.10455

Adapted for gesture and emotion recognition from pose sequences.
"""

import torch
import torch.nn as nn
import math
from einops import rearrange, repeat


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    Adds position information to token embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SpatialEncoding(nn.Module):
    """
    Spatial encoding for joint positions.
    Each joint gets a learnable embedding.
    """
    def __init__(self, num_joints: int, d_model: int):
        super().__init__()
        self.spatial_embed = nn.Parameter(torch.randn(1, num_joints, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, num_joints, d_model)
        """
        return x + self.spatial_embed


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor of same shape as input
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)  # (batch, heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2)  # (batch, seq_len, heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer encoder block.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class PoseFormer(nn.Module):
    """
    Complete PoseFormer model for gesture and emotion recognition.
    
    Architecture:
        Input Poses -> Joint Embedding -> Spatial Encoding -> 
        Positional Encoding -> Transformer Blocks -> 
        Global Pooling -> Classification Head
    """
    def __init__(
        self,
        num_classes: int,
        num_joints: int = 25,
        num_frames: int = 300,
        input_dim: int = 3,  # (x, y, confidence) or (x, y, z)
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        pooling: str = 'cls'  # 'cls', 'mean', or 'max'
    ):
        super().__init__()
        
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.d_model = d_model
        self.pooling = pooling
        
        # Input embedding: project joint coordinates to d_model
        self.joint_embed = nn.Linear(input_dim, d_model)
        
        # Spatial encoding for joints
        self.spatial_encoding = SpatialEncoding(num_joints, d_model)
        
        # Positional encoding for temporal information
        self.pos_encoding = PositionalEncoding(d_model, max_len=num_frames * num_joints, dropout=dropout)
        
        # CLS token for classification (if using cls pooling)
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, frames, joints, persons)
               or (batch, frames, joints, channels)
        
        Returns:
            Output logits (batch, num_classes)
        """
        # Handle different input formats
        if x.dim() == 5:  # (batch, channels, frames, joints, persons)
            batch_size, channels, frames, joints, persons = x.shape
            # Take only first person and rearrange
            x = x[:, :, :, :, 0]  # (batch, channels, frames, joints)
            x = x.permute(0, 2, 3, 1)  # (batch, frames, joints, channels)
        else:  # (batch, frames, joints, channels)
            batch_size, frames, joints, channels = x.shape
        
        # Embed each joint's coordinates
        x = self.joint_embed(x)  # (batch, frames, joints, d_model)
        
        # Flatten frames and joints into sequence
        x = rearrange(x, 'b t j d -> b (t j) d')  # (batch, frames*joints, d_model)
        
        # Add CLS token if using cls pooling
        if self.pooling == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Normalize
        x = self.norm(x)
        
        # Pooling
        if self.pooling == 'cls':
            x = x[:, 0]  # Take CLS token
        elif self.pooling == 'mean':
            x = x.mean(dim=1)  # Average pooling
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]  # Max pooling
        
        # Classification
        logits = self.fc(x)
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        if x.dim() == 5:
            batch_size, channels, frames, joints, persons = x.shape
            x = x[:, :, :, :, 0]
            x = x.permute(0, 2, 3, 1)
        else:
            batch_size, frames, joints, channels = x.shape
        
        x = self.joint_embed(x)
        x = rearrange(x, 'b t j d -> b (t j) d')
        
        if self.pooling == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
            x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.pos_encoding(x)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        x = self.norm(x)
        
        if self.pooling == 'cls':
            features = x[:, 0]
        elif self.pooling == 'mean':
            features = x.mean(dim=1)
        elif self.pooling == 'max':
            features = x.max(dim=1)[0]
        
        return features


def create_poseformer_model(
    num_classes: int,
    num_joints: int = 25,
    num_frames: int = 300,
    d_model: int = 256,
    num_layers: int = 6,
    dropout: float = 0.1
) -> PoseFormer:
    """
    Factory function to create PoseFormer model.
    
    Args:
        num_classes: Number of output classes
        num_joints: Number of skeleton joints
        num_frames: Number of frames in sequence
        d_model: Model dimension
        num_layers: Number of transformer layers
        dropout: Dropout rate
    
    Returns:
        Initialized PoseFormer model
    """
    model = PoseFormer(
        num_classes=num_classes,
        num_joints=num_joints,
        num_frames=num_frames,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
        pooling='cls'
    )
    return model


if __name__ == "__main__":
    # Test the model
    model = create_poseformer_model(num_classes=15, num_joints=25, num_frames=300)
    
    # Test with different input formats
    # Format 1: (batch, channels, frames, joints, persons)
    dummy_input1 = torch.randn(4, 3, 300, 25, 1)
    output1 = model(dummy_input1)
    
    # Format 2: (batch, frames, joints, channels)
    dummy_input2 = torch.randn(4, 300, 25, 3)
    output2 = model(dummy_input2)
    
    print(f"Input format 1: {dummy_input1.shape} -> Output: {output1.shape}")
    print(f"Input format 2: {dummy_input2.shape} -> Output: {output2.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")