"""
Vision Transformer (ViT) model for continual learning experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import get_activation, get_normalization, PatchEmbedding

class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Store attention maps
        self.attention_maps = None

    def forward(self, x, store_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        if store_attention:
            self.attention_maps = attn.detach().clone()
            
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def get_attention_maps(self):
        return self.attention_maps


class MLP(nn.Module):
    """MLP module with configurable activation."""
    def __init__(self, in_features, hidden_features, out_features, 
                 activation='gelu', drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = get_activation(activation)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with customizable components."""
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=True, drop=0., 
                 attn_drop=0., activation='gelu', normalization='layer'):
        super().__init__()
        self.norm1 = get_normalization(normalization, dim)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = get_normalization(normalization, dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, 
                      activation=activation, drop=drop)
        
        # Activations storage
        self.stored_activations = {}

    def forward(self, x, store_activations=False):
        # Store input
        if store_activations:
            self.stored_activations['input'] = x.detach().clone()
        
        # Self-attention
        norm_x = self.norm1(x)
        if store_activations:
            self.stored_activations['norm1'] = norm_x.detach().clone()
            
        attn_out = self.attn(norm_x, store_attention=store_activations)
        if store_activations:
            self.stored_activations['attn_out'] = attn_out.detach().clone()
            self.stored_activations['attn_maps'] = self.attn.get_attention_maps()
            
        x = x + attn_out
        if store_activations:
            self.stored_activations['post_attn'] = x.detach().clone()
        
        # MLP
        norm_x = self.norm2(x)
        if store_activations:
            self.stored_activations['norm2'] = norm_x.detach().clone()
            
        mlp_out = self.mlp(norm_x)
        if store_activations:
            self.stored_activations['mlp_out'] = mlp_out.detach().clone()
            
        x = x + mlp_out
        if store_activations:
            self.stored_activations['output'] = x.detach().clone()
            
        return x
    
    def get_activations(self):
        return self.stored_activations


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model with configurable architecture.
    """
    def __init__(self, 
                 img_size=32, 
                 patch_size=4, 
                 in_channels=3, 
                 num_classes=10, 
                 embed_dim=192,
                 depth=12, 
                 n_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop_rate=0.1,
                 attn_drop_rate=0.0,
                 activation='gelu',
                 normalization='layer',
                 record_activations=False):
        """
        Initialize Vision Transformer.
        
        Parameters:
            img_size (int): Input image size
            patch_size (int): Patch size for splitting image
            in_channels (int): Number of image channels
            num_classes (int): Number of output classes
            embed_dim (int): Embedding dimension
            depth (int): Number of transformer blocks
            n_heads (int): Number of attention heads
            mlp_ratio (float): Ratio for MLP hidden dimension
            qkv_bias (bool): Whether to use bias in QKV projection
            drop_rate (float): Dropout rate
            attn_drop_rate (float): Attention dropout rate
            activation (str): Activation function to use
            normalization (str): Normalization method to use
            record_activations (bool): Whether to store activations
        """
        super().__init__()
        self.record_activations = record_activations
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, qkv_bias, 
                           drop_rate, attn_drop_rate, activation, normalization)
            for _ in range(depth)
        ])

        # Final normalization and classifier head
        self.norm = get_normalization(normalization, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()
        
        # Activation storage
        self.stored_activations = {}

    def _init_weights(self):
        # Initialize position embedding and class token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, store_activations=False):
        """
        Forward pass with optional activation storage.
        
        Parameters:
            x (torch.Tensor): Input images [batch_size, channels, height, width]
            store_activations (bool): Whether to store activations
            
        Returns:
            torch.Tensor: Output logits
            dict (optional): Hidden activations if record_activations=True
        """
        # Should we store activations for this pass?
        should_store = store_activations or self.record_activations
        activations = {} if should_store else None
        
        if should_store:
            activations['input'] = x.detach().clone()
        
        # Patch embedding
        x = self.patch_embed(x)
        if should_store:
            activations['patch_embed'] = x.detach().clone()
        
        # Add class token
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        if should_store:
            activations['pos_embed'] = x.detach().clone()
            
        x = self.pos_drop(x)

        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x, store_activations=should_store)
            if should_store:
                activations[f'block_{i}_output'] = x.detach().clone()
                if store_activations:
                    block_acts = block.get_activations()
                    for k, v in block_acts.items():
                        activations[f'block_{i}_{k}'] = v
        
        # Final normalization
        x = self.norm(x)
        if should_store:
            activations['final_norm'] = x.detach().clone()
        
        # Extract class token and classify
        x = x[:, 0]  # Use only the cls token for classification
        if should_store:
            activations['cls_token'] = x.detach().clone()
            
        x = self.head(x)
        if should_store:
            activations['output'] = x.detach().clone()
            self.stored_activations = activations
            return x, activations
            
        return x
    
    def get_activations(self):
        """Returns stored activations from the last forward pass"""
        return self.stored_activations