"""
Rotary Position Embedding (RoPE) for Time Conditioning

시간 정보를 attention의 query/key에 회전 변환으로 주입합니다.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RoPE(nn.Module):
    """
    Rotary Position Embedding for time conditioning.
    
    기존 위치 정보 대신 시간(t) 정보를 RoPE로 인코딩합니다.
    
    Args:
        dim: Embedding dimension (must be even)
        max_freq: Maximum frequency for sinusoidal encoding
        num_freqs: Number of frequency bands (default: dim // 2)
    """
    
    def __init__(
        self,
        dim: int,
        max_freq: float = 10000.0,
        num_freqs: Optional[int] = None,
    ):
        super().__init__()
        
        assert dim % 2 == 0, f"Dimension must be even, got {dim}"
        
        self.dim = dim
        self.max_freq = max_freq
        self.num_freqs = num_freqs if num_freqs is not None else dim // 2
        
        # Precompute frequency bands
        freqs = torch.exp(
            torch.arange(0, self.num_freqs, dtype=torch.float32) * 
            (-math.log(max_freq) / self.num_freqs)
        )
        self.register_buffer('freqs', freqs)
        
    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotation matrices (cos, sin) for given timesteps.
        
        Args:
            t: Timesteps [B] or [B, 1], values in [0, 1]
            
        Returns:
            cos: Cosine components [B, dim]
            sin: Sine components [B, dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]
            
        # Scale timesteps
        t = t * self.max_freq
        
        # Compute angles: [B, num_freqs]
        angles = t * self.freqs.unsqueeze(0)
        
        # Compute cos and sin
        cos = torch.cos(angles)  # [B, num_freqs]
        sin = torch.sin(angles)  # [B, num_freqs]
        
        # Repeat for full dimension (each freq applies to 2 dims)
        cos = torch.cat([cos, cos], dim=-1)  # [B, dim]
        sin = torch.cat([sin, sin], dim=-1)  # [B, dim]
        
        return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding to input tensor.
    
    Args:
        x: Input tensor [B, N, dim] or [B, heads, N, head_dim]
        cos: Cosine components [B, dim] or [B, 1, 1, head_dim]
        sin: Sine components [B, dim] or [B, 1, 1, head_dim]
        
    Returns:
        Rotated tensor with same shape as input
    """
    # Handle different input shapes
    if x.dim() == 3:
        # [B, N, dim] -> expand cos/sin to [B, 1, dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif x.dim() == 4:
        # [B, heads, N, head_dim] -> expand cos/sin to [B, 1, 1, head_dim]
        cos = cos.unsqueeze(1).unsqueeze(1)
        sin = sin.unsqueeze(1).unsqueeze(1)
    
    # Split x into two halves for rotation
    dim = x.shape[-1]
    x1, x2 = x[..., :dim//2], x[..., dim//2:]
    
    # Apply rotation
    # Rotation formula: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    cos1, cos2 = cos[..., :dim//2], cos[..., dim//2:]
    sin1, sin2 = sin[..., :dim//2], sin[..., dim//2:]
    
    rotated_x1 = x1 * cos1 - x2 * sin1
    rotated_x2 = x1 * sin2 + x2 * cos2
    
    return torch.cat([rotated_x1, rotated_x2], dim=-1)


class TimeEmbedding(nn.Module):
    """
    Time embedding module combining sinusoidal embedding with MLP.
    
    Args:
        dim: Output embedding dimension
        hidden_dim: Hidden layer dimension (default: 4 * dim)
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.dim = dim
        hidden_dim = hidden_dim or dim * 4
        
        # Sinusoidal embedding
        self.rope = RoPE(dim)
        
        # MLP to process embedding
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time embedding.
        
        Args:
            t: Timesteps [B], values in [0, 1]
            
        Returns:
            Time embedding [B, dim]
        """
        # Get sinusoidal embedding via RoPE
        cos, sin = self.rope(t)
        
        # Combine cos and sin as initial embedding
        # Use cos as the base embedding
        emb = cos
        
        # Process through MLP
        emb = self.mlp(emb)
        
        return emb


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal time embedding (alternative to RoPE-based).
    
    Args:
        dim: Embedding dimension
        max_period: Maximum period for frequencies
    """
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps [B], values in [0, 1]
            
        Returns:
            Embedding [B, dim]
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * 
            torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        
        # Scale t to [0, max_period]
        t = t.unsqueeze(-1) * self.max_period
        args = t * freqs.unsqueeze(0)
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return embedding


if __name__ == "__main__":
    """
    RoPE Time Embedding 테스트
    """
    print("=" * 60)
    print("RoPE Time Embedding Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. RoPE 테스트
    print("\n[1] RoPE 테스트...")
    rope = RoPE(dim=256).to(device)
    t = torch.rand(4, device=device)  # [B]
    cos, sin = rope(t)
    print(f"    Input t shape: {t.shape}")
    print(f"    Cos shape: {cos.shape}")
    print(f"    Sin shape: {sin.shape}")
    
    # 2. apply_rope 테스트
    print("\n[2] apply_rope 테스트...")
    x = torch.randn(4, 16, 256, device=device)  # [B, N, dim]
    rotated = apply_rope(x, cos, sin)
    print(f"    Input shape: {x.shape}")
    print(f"    Rotated shape: {rotated.shape}")
    
    # 3. TimeEmbedding 테스트
    print("\n[3] TimeEmbedding 테스트...")
    time_emb = TimeEmbedding(dim=256).to(device)
    t = torch.rand(4, device=device)
    emb = time_emb(t)
    print(f"    Input t shape: {t.shape}")
    print(f"    Embedding shape: {emb.shape}")
    
    # 4. SinusoidalTimeEmbedding 테스트
    print("\n[4] SinusoidalTimeEmbedding 테스트...")
    sin_emb = SinusoidalTimeEmbedding(dim=256).to(device)
    emb2 = sin_emb(t)
    print(f"    Embedding shape: {emb2.shape}")
    
    print("\n" + "=" * 60)
    print("모든 테스트 완료!")
    print("=" * 60)

