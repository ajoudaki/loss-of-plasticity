import torch
import torch.nn as nn
from .layers import get_activation, get_normalization

class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.scale = head_dim ** -0.5

        self.layers = nn.ModuleDict({
            'qkv': nn.Linear(dim, dim * 3, bias=qkv_bias),
            'attn_drop': nn.Dropout(attn_drop),
            'proj': nn.Linear(dim, dim),
            'proj_drop': nn.Dropout(proj_drop)
        })

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.layers['qkv'](x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.layers['attn_drop'](attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.layers['proj'](x)
        x = self.layers['proj_drop'](x)
        return x


class TransformerMLP(nn.Module):
    """MLP module with activation."""
    def __init__(self, in_features, hidden_features, out_features, 
                 activation='gelu', drop=0.):
        super().__init__()
        
        self.layers = nn.ModuleDict({
            'fc1': nn.Linear(in_features, hidden_features),
            'act': get_activation(activation),
            'drop1': nn.Dropout(drop) if drop > 0 else nn.Identity(),
            'fc2': nn.Linear(hidden_features, out_features),
            'drop2': nn.Dropout(drop) if drop > 0 else nn.Identity()
        })

    def forward(self, x):
        x = self.layers['fc1'](x)
        x = self.layers['act'](x)
        x = self.layers['drop1'](x)
        x = self.layers['fc2'](x)
        x = self.layers['drop2'](x)
        return x


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding for Vision Transformer."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.layers = nn.ModuleDict({
            'proj': nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
        })

    def forward(self, x):
        x = self.layers['proj'](x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with components."""
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=True, drop=0., 
                 attn_drop=0., activation='gelu', normalization='layer',
                 normalization_affine=True):
        super().__init__()
        
        self.layers = nn.ModuleDict({
            'norm1': get_normalization(normalization, dim, affine=normalization_affine),
            'attn': Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, 
                             attn_drop=attn_drop, proj_drop=drop),
            'norm2': get_normalization(normalization, dim, affine=normalization_affine),
            'mlp': TransformerMLP(dim, int(dim * mlp_ratio), dim, 
                       activation=activation, drop=drop)
        })

    def forward(self, x):
        norm_x = self.layers['norm1'](x)
        attn_out = self.layers['attn'](norm_x)
        x = x + attn_out
        
        norm_x = self.layers['norm2'](x)
        mlp_out = self.layers['mlp'](norm_x)
        x = x + mlp_out
            
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) model."""
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
                 dropout_p=0.1,
                 attn_drop_rate=0.0,
                 activation='gelu',
                 normalization='layer',
                 normalization_affine=True):
        super().__init__()
        
        self.layers = nn.ModuleDict()
        
        self.layers['patch_embed'] = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.layers['patch_embed'].n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        
        self.layers['pos_drop'] = nn.Dropout(dropout_p)

        for i in range(depth):
            self.layers[f'block_{i}'] = TransformerBlock(
                embed_dim, n_heads, mlp_ratio, qkv_bias, 
                dropout_p, attn_drop_rate, activation, normalization,
                normalization_affine=normalization_affine
            )

        self.layers['norm'] = get_normalization(normalization, embed_dim, affine=normalization_affine)
        self.layers['out'] = nn.Linear(embed_dim, num_classes)

        self._init_weights()
        self.depth = depth

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.layers['patch_embed'](x)
        
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = x + self.pos_embed
        x = self.layers['pos_drop'](x)

        for i in range(self.depth):
            x = self.layers[f'block_{i}'](x)
        
        x = self.layers['norm'](x)
        x = x[:, 0]  # Use CLS token for classification
        x = self.layers['out'](x)
            
        return x