"""
AdaIN-Zero: Adaptive Instance Normalization with Zero-Initialization

DiT에서 time conditioning을 주입하기 위한 AdaIN-Zero 모듈입니다.
Zero-initialization을 통해 학습 초기 안정성을 보장합니다.
"""

import torch
import torch.nn as nn
from typing import Optional


class AdaINZero(nn.Module):
    """
    Adaptive Instance Normalization with Zero-Initialization.
    
    Time embedding을 scale과 shift로 변환하여 feature에 적용합니다.
    출력 projection이 zero-initialized되어 학습 초기에는 identity mapping에 가깝습니다.
    
    Args:
        dim: Feature dimension
        time_dim: Time embedding dimension (default: same as dim)
    """
    
    def __init__(
        self,
        dim: int,
        time_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.dim = dim
        time_dim = time_dim or dim
        
        # LayerNorm for input normalization
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        
        # MLP to produce scale and shift from time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2),  # scale and shift
        )
        
        # Zero-initialize the final linear layer
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)
        
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply AdaIN-Zero conditioning.
        
        Args:
            x: Input features [B, N, dim] or [B, dim, H, W]
            time_emb: Time embedding [B, time_dim]
            
        Returns:
            Conditioned features with same shape as input
        """
        # Compute scale and shift
        scale_shift = self.time_mlp(time_emb)  # [B, dim * 2]
        scale, shift = scale_shift.chunk(2, dim=-1)  # [B, dim] each
        
        # Handle different input shapes
        if x.dim() == 4:
            # [B, dim, H, W] -> [B, H, W, dim]
            x = x.permute(0, 2, 3, 1)
            is_conv = True
        else:
            is_conv = False
            
        # Normalize
        x = self.norm(x)
        
        # Reshape scale/shift for broadcasting
        if x.dim() == 3:
            # [B, N, dim]
            scale = scale.unsqueeze(1)  # [B, 1, dim]
            shift = shift.unsqueeze(1)  # [B, 1, dim]
        elif x.dim() == 4:
            # [B, H, W, dim]
            scale = scale.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, dim]
            shift = shift.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, dim]
            
        # Apply scale and shift: x = x * (1 + scale) + shift
        x = x * (1.0 + scale) + shift
        
        if is_conv:
            # [B, H, W, dim] -> [B, dim, H, W]
            x = x.permute(0, 3, 1, 2)
            
        return x


class AdaINZeroBlock(nn.Module):
    """
    A complete AdaIN-Zero block with optional residual connection.
    
    Args:
        dim: Feature dimension
        time_dim: Time embedding dimension
        use_residual: Whether to use residual connection
    """
    
    def __init__(
        self,
        dim: int,
        time_dim: Optional[int] = None,
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.use_residual = use_residual
        self.adain = AdaINZero(dim, time_dim)
        
        # Optional feedforward
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
        # Zero-initialize output
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)
        
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features [B, N, dim]
            time_emb: Time embedding [B, time_dim]
            
        Returns:
            Output features [B, N, dim]
        """
        # AdaIN conditioning
        h = self.adain(x, time_emb)
        
        # Feedforward
        h = self.ff(h)
        
        # Residual connection
        if self.use_residual:
            return x + h
        return h


class ModulatedLayerNorm(nn.Module):
    """
    LayerNorm modulated by time embedding (alternative to AdaIN-Zero).
    
    Args:
        dim: Feature dimension
        time_dim: Time embedding dimension
    """
    
    def __init__(self, dim: int, time_dim: Optional[int] = None):
        super().__init__()
        
        time_dim = time_dim or dim
        
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        
        # Learnable base scale and shift
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
        # Time-dependent modulation
        self.time_proj = nn.Linear(time_dim, dim * 2)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, N, dim]
            time_emb: Time embedding [B, time_dim]
            
        Returns:
            Modulated output [B, N, dim]
        """
        # Normalize
        x = self.norm(x)
        
        # Base affine
        x = x * self.gamma + self.beta
        
        # Time modulation
        mod = self.time_proj(time_emb)  # [B, dim * 2]
        scale, shift = mod.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)  # [B, 1, dim]
        shift = shift.unsqueeze(1)
        
        x = x * (1.0 + scale) + shift
        
        return x


if __name__ == "__main__":
    """
    AdaIN-Zero 테스트
    """
    print("=" * 60)
    print("AdaIN-Zero Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. AdaINZero 테스트 (sequence input)
    print("\n[1] AdaINZero 테스트 (sequence)...")
    adain = AdaINZero(dim=256, time_dim=256).to(device)
    x = torch.randn(4, 16, 256, device=device)  # [B, N, dim]
    t_emb = torch.randn(4, 256, device=device)  # [B, time_dim]
    out = adain(x, t_emb)
    print(f"    Input shape: {x.shape}")
    print(f"    Time emb shape: {t_emb.shape}")
    print(f"    Output shape: {out.shape}")
    
    # 2. AdaINZero 테스트 (conv input)
    print("\n[2] AdaINZero 테스트 (conv)...")
    x_conv = torch.randn(4, 256, 16, 16, device=device)  # [B, C, H, W]
    out_conv = adain(x_conv, t_emb)
    print(f"    Input shape: {x_conv.shape}")
    print(f"    Output shape: {out_conv.shape}")
    
    # 3. AdaINZeroBlock 테스트
    print("\n[3] AdaINZeroBlock 테스트...")
    block = AdaINZeroBlock(dim=256, time_dim=256).to(device)
    out_block = block(x, t_emb)
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {out_block.shape}")
    
    # 4. Zero-initialization 확인
    print("\n[4] Zero-initialization 확인...")
    adain_fresh = AdaINZero(dim=256, time_dim=256)
    # 초기 상태에서 scale=0, shift=0이므로 출력은 normalized input과 같아야 함
    x_test = torch.randn(2, 8, 256)
    t_test = torch.randn(2, 256)
    out_test = adain_fresh(x_test, t_test)
    norm_x = nn.LayerNorm(256, elementwise_affine=False)(x_test)
    diff = (out_test - norm_x).abs().max().item()
    print(f"    Max diff from normalized input: {diff:.6f}")
    print(f"    Zero-init working: {diff < 1e-5}")
    
    print("\n" + "=" * 60)
    print("모든 테스트 완료!")
    print("=" * 60)

