"""
DiT (Diffusion Transformer) for 64x64 Image Generation

64x64 이미지를 입력받아 64x64 velocity를 예측하는 Transformer 모델입니다.
AdaIN-Zero와 RoPE time conditioning을 사용합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

try:
    from .adain_zero import AdaINZero
except ImportError:
    from adain_zero import AdaINZero


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as JiT/DiT implementation.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class VisionRotaryEmbedding(nn.Module):
    """
    2D Rotary Position Embedding for Vision Transformers.
    Applies RoPE based on 2D spatial positions of patches.
    
    Reference: JiT, RoFormer, LLaMA
    """
    def __init__(self, dim: int, grid_size: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        
        # Compute frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('freqs', freqs)
        
        # Precompute 2D position embeddings
        self._build_rope_cache(grid_size)
        
    def _build_rope_cache(self, grid_size: int):
        """Build cos/sin cache for 2D grid positions."""
        # Create 2D grid positions
        y = torch.arange(grid_size)
        x = torch.arange(grid_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # Flatten to sequence
        pos_y = grid_y.flatten().float()  # [grid_size^2]
        pos_x = grid_x.flatten().float()  # [grid_size^2]
        
        # Compute angles for each dimension
        # Use half dimensions for x, half for y
        half_dim = self.dim // 2
        
        freqs_y = pos_y.unsqueeze(-1) * self.freqs[:half_dim//2].unsqueeze(0)  # [N, dim//4]
        freqs_x = pos_x.unsqueeze(-1) * self.freqs[:half_dim//2].unsqueeze(0)  # [N, dim//4]
        
        # Concatenate x and y frequencies
        freqs = torch.cat([freqs_y, freqs_x], dim=-1)  # [N, dim//2]
        
        # Compute cos and sin
        cos = torch.cos(freqs)  # [N, dim//2]
        sin = torch.sin(freqs)  # [N, dim//2]
        
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D RoPE to input tensor.
        
        Args:
            x: Input tensor [B, num_heads, N, head_dim]
            
        Returns:
            Rotated tensor [B, num_heads, N, head_dim]
        """
        B, H, N, D = x.shape
        
        # Get cached cos/sin
        cos = self.cos_cached[:N]  # [N, dim//2]
        sin = self.sin_cached[:N]  # [N, dim//2]
        
        # Expand for batch and heads
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, N, dim//2]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, N, dim//2]
        
        # Split x into two halves
        x1, x2 = x[..., :D//2], x[..., D//2:]
        
        # Apply rotation
        # [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        return torch.cat([rotated_x1, rotated_x2], dim=-1)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding (Linear Embed).
    
    Args:
        img_size: Input image size (default: 64)
        patch_size: Patch size (default: 4)
        in_channels: Number of input channels (default: 3)
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 512,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 64/4 = 16 -> 16x16 = 256 patches
        self.grid_size = img_size // patch_size
        
        # Linear projection (Conv2d with kernel=patch_size, stride=patch_size)
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            Patch embeddings [B, num_patches, embed_dim]
        """
        # [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.proj(x)
        # [B, embed_dim, H/P, W/P] -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x


class Unpatchify(nn.Module):
    """
    Patch embeddings back to image (Linear Predict).
    
    Args:
        img_size: Output image size (default: 64)
        patch_size: Patch size (default: 4)
        out_channels: Number of output channels (default: 3)
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        out_channels: int = 3,
        embed_dim: int = 512,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.grid_size = img_size // patch_size
        
        # Linear prediction: embed_dim -> patch_size * patch_size * out_channels
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings [B, num_patches, embed_dim]
            
        Returns:
            Image [B, C, H, W]
        """
        B, N, D = x.shape
        C = self.out_channels
        P = self.patch_size
        G = self.grid_size
        
        # [B, N, D] -> [B, N, P*P*C]
        x = self.proj(x)
        
        # Reshape: [B, G, G, P, P, C] (same as JiT unpatchify)
        x = x.reshape(B, G, G, P, P, C)
        
        # einsum 'nhwpqc->nchpwq' equivalent:
        # [B, G, G, P, P, C] -> [B, C, G, P, G, P] -> [B, C, H, W]
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, self.img_size, self.img_size)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with 2D RoPE.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(
        self, 
        x: torch.Tensor, 
        rope: Optional[VisionRotaryEmbedding] = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply 2D RoPE to Q and K (spatial position encoding)
        if rope is not None:
            q = rope(q)
            k = rope(k)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU."""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """
    Transformer Block with AdaIN-Zero conditioning and 2D RoPE.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_dim: Optional[int] = None,
    ):
        super().__init__()
        
        time_dim = time_dim or dim
        
        self.adain1 = AdaINZero(dim, time_dim)
        self.adain2 = AdaINZero(dim, time_dim)
        
        self.attn = MultiHeadAttention(dim=dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout)
        self.ff = FeedForward(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)
        
        # Zero-init
        nn.init.zeros_(self.attn.proj.weight)
        nn.init.zeros_(self.attn.proj.bias)
        nn.init.zeros_(self.ff.net[-2].weight)
        nn.init.zeros_(self.ff.net[-2].bias)
        
    def forward(
        self, 
        x: torch.Tensor, 
        time_emb: torch.Tensor,
        rope: Optional[VisionRotaryEmbedding] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.adain1(x, time_emb), rope=rope)
        x = x + self.ff(self.adain2(x, time_emb))
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer for 64x64 Image Generation.
    
    Architecture:
        Input (64x64) -> Patchify -> Linear Embed -> Transformer Blocks -> Linear Predict -> Output (64x64)
    
    Args:
        img_size: Image size (default: 64)
        patch_size: Patch size (default: 4)
        in_channels: Input channels (default: 3)
        out_channels: Output channels (default: 3)
        embed_dim: Embedding dimension (default: 512)
        depth: Number of transformer blocks (default: 12)
        num_heads: Number of attention heads (default: 8)
        mlp_ratio: MLP hidden dim ratio (default: 4.0)
        dropout: Dropout rate (default: 0.0)
        use_rope: Use 2D RoPE for spatial position encoding (default: True)
    """
    
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_rope = use_rope
        
        # Patchify + Linear Embed
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        grid_size = img_size // patch_size
        
        # Positional embedding (learnable) - can be used with or without RoPE
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 2D RoPE for spatial positions (like JiT)
        if use_rope:
            head_dim = embed_dim // num_heads
            self.rope = VisionRotaryEmbedding(dim=head_dim, grid_size=grid_size)
        else:
            self.rope = None
        
        # Time embedding (sinusoidal + MLP, like JiT)
        self.time_embed = TimestepEmbedder(embed_dim)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, time_dim=embed_dim)
            for _ in range(depth)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Linear Predict + Unpatchify
        self.unpatchify = Unpatchify(img_size, patch_size, out_channels, embed_dim)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict x (clean image) from noisy image and timestep.
        
        Args:
            x: Noisy image [B, 3, 64, 64]
            t: Timesteps [B], values in [0, 1]
            
        Returns:
            Predicted clean image [B, 3, 64, 64]
        """
        # Patchify + Linear Embed + Position
        x = self.patch_embed(x)  # [B, 256, embed_dim] (64/4 = 16, 16x16 = 256 patches)
        x = x + self.pos_embed
        
        # Time embedding
        time_emb = self.time_embed(t)
        
        # Transformer Blocks with 2D RoPE
        for block in self.blocks:
            x = block(x, time_emb, rope=self.rope)
            
        # Final norm
        x = self.final_norm(x)
        
        # Linear Predict + Unpatchify
        x_pred = self.unpatchify(x)  # [B, 3, 64, 64]
        
        return x_pred


def build_dit(
    img_size: int = 64,
    patch_size: int = 4,
    model_size: str = "base",
    **kwargs,
) -> DiT:
    """
    Build DiT model with predefined configurations.
    
    Args:
        img_size: Image size (default: 64)
        patch_size: Patch size (default: 4)
        model_size: 'small', 'base', 'large'
        
    Returns:
        DiT model
    """
    configs = {
        "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        "base": {"embed_dim": 512, "depth": 12, "num_heads": 8},
        "large": {"embed_dim": 768, "depth": 24, "num_heads": 12},
    }
    
    config = configs.get(model_size, configs["base"])
    config.update(kwargs)
    
    return DiT(img_size=img_size, patch_size=patch_size, **config)


if __name__ == "__main__":
    """
    DiT 64x64 테스트
    """
    print("=" * 60)
    print("DiT 64x64 Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 모델 생성
    print("\n[1] 모델 생성...")
    model = DiT(
        img_size=64,
        patch_size=4,
        embed_dim=384,
        depth=6,
        num_heads=6,
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    Parameters: {params:.2f}M")
    print(f"    Num patches: {model.num_patches} (64/4 = 16 -> 16x16)")
    
    # Forward
    print("\n[2] Forward 테스트...")
    x = torch.randn(2, 3, 64, 64, device=device)
    t = torch.rand(2, device=device)
    
    with torch.no_grad():
        x_pred = model(x, t)
    
    print(f"    Input: {x.shape}")
    print(f"    Time: {t.shape}")
    print(f"    Output (x-pred): {x_pred.shape}")
    
    # build_dit
    print("\n[3] build_dit 테스트...")
    for size in ["small", "base", "large"]:
        m = build_dit(model_size=size)
        p = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"    {size}: {p:.2f}M")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
