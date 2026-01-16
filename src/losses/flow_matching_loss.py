"""
Flow Matching Loss for Rectified Flow (JiT-style)

Reference: https://github.com/LTH14/JiT

Key features:
- Logit-normal t sampling (P_mean, P_std) for stable training
- noise_scale parameter
- v-prediction with proper numerical stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FlowMatchingLoss(nn.Module):
    """
    JiT-style Flow Matching Loss for Rectified Flow training.
    
    핵심 기법:
    1. Logit-normal t 샘플링: t 분포를 중간 영역에 집중시켜 안정성 향상
    2. noise_scale: 노이즈 스케일 조절
    3. t_eps: 수치 안정성을 위한 클램핑
    
    Args:
        P_mean: Logit-normal 분포의 평균 (default: -0.8)
        P_std: Logit-normal 분포의 표준편차 (default: 0.8)
        noise_scale: 노이즈 스케일 (default: 1.0)
        t_eps: 수치 안정성을 위한 epsilon (default: 1e-5)
    """
    
    def __init__(
        self, 
        P_mean: float = -0.8, 
        P_std: float = 0.8,
        noise_scale: float = 1.0,
        t_eps: float = 1e-5,
    ):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        
    def sample_t(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Logit-normal distribution으로 t 샘플링.
        
        P_mean=-0.8, P_std=0.8 일 때:
        - t의 mode ≈ 0.31 (중간-낮은 영역)
        - t=1 근처 샘플이 적어 수치 안정성 향상
        """
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)
        
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        JiT-style Flow Matching loss 계산.
        
        Args:
            model: DiT 모델 (x_pred 예측)
            x: Clean images [B, C, H, W], range [-1, 1]
            t: Timesteps [B] (optional, None이면 logit-normal 샘플링)
            
        Returns:
            loss: Scalar loss value
            info: Dictionary with additional info
        """
        B = x.shape[0]
        device = x.device
        
        # Sample timestep t ~ Logit-normal(P_mean, P_std)
        if t is None:
            t = self.sample_t(B, device)
        
        t_expand = t.view(B, 1, 1, 1)
        
        # Sample noise e ~ N(0, noise_scale^2 * I)
        e = torch.randn_like(x) * self.noise_scale
        
        # Interpolate: z = t * x + (1 - t) * e
        z = t_expand * x + (1 - t_expand) * e
        
        # Target velocity: v = (x - z) / (1 - t)
        v_target = (x - z) / (1 - t_expand).clamp(min=self.t_eps)
        
        # Model prediction: x_pred = net(z, t)
        x_pred = model(z, t)
        
        # Predicted velocity: v_pred = (x_pred - z) / (1 - t)
        v_pred = (x_pred - z) / (1 - t_expand).clamp(min=self.t_eps)
        
        # L2 Loss: ||v_pred - v_target||^2
        # Per-sample loss, then mean
        loss = ((v_pred - v_target) ** 2).mean(dim=(1, 2, 3)).mean()
        
        # Additional info for logging
        info = {
            "loss": loss.item(),
            "t_mean": t.mean().item(),
            "t_std": t.std().item(),
            "v_target_norm": v_target.norm().item() / B,
            "v_pred_norm": v_pred.norm().item() / B,
            "x_pred_range": (x_pred.min().item(), x_pred.max().item()),
        }
        
        return loss, info


def flow_matching_loss(
    model: nn.Module,
    x: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    P_mean: float = -0.8,
    P_std: float = 0.8,
    noise_scale: float = 1.0,
    t_eps: float = 1e-5,
) -> torch.Tensor:
    """
    Functional interface for JiT-style flow matching loss.
    
    Args:
        model: DiT model
        x: Clean images [B, C, H, W]
        t: Timesteps [B] (optional)
        P_mean: Logit-normal mean
        P_std: Logit-normal std
        noise_scale: Noise scale
        t_eps: Numerical stability epsilon
        
    Returns:
        loss: Scalar loss
    """
    B = x.shape[0]
    device = x.device
    
    # Logit-normal sampling
    if t is None:
        z = torch.randn(B, device=device) * P_std + P_mean
        t = torch.sigmoid(z)
    
    t_expand = t.view(B, 1, 1, 1)
    e = torch.randn_like(x) * noise_scale
    
    # z = t * x + (1 - t) * e
    z = t_expand * x + (1 - t_expand) * e
    
    # v_target and v_pred
    v_target = (x - z) / (1 - t_expand).clamp(min=t_eps)
    x_pred = model(z, t)
    v_pred = (x_pred - z) / (1 - t_expand).clamp(min=t_eps)
    
    # L2 loss
    loss = ((v_pred - v_target) ** 2).mean(dim=(1, 2, 3)).mean()
    
    return loss


if __name__ == "__main__":
    """
    Flow Matching Loss 테스트
    """
    import sys
    sys.path.insert(0, '/workspace/CelebGen')
    from src.models import build_model
    
    print("=" * 60)
    print("Flow Matching Loss Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 모델 생성
    print("\n[1] 모델 생성...")
    model = build_model(model_size="small", depth=6).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    Parameters: {params:.2f}M")
    
    # Loss 모듈 생성
    print("\n[2] FlowMatchingLoss 테스트...")
    criterion = FlowMatchingLoss()
    
    x = torch.randn(4, 3, 64, 64, device=device)  # Clean images
    loss, info = criterion(model, x)
    
    print(f"    Loss: {loss.item():.4f}")
    print(f"    t_mean: {info['t_mean']:.4f}")
    print(f"    v_target_norm: {info['v_target_norm']:.4f}")
    print(f"    v_pred_norm: {info['v_pred_norm']:.4f}")
    
    # Functional interface 테스트
    print("\n[3] flow_matching_loss 함수 테스트...")
    loss2 = flow_matching_loss(model, x)
    print(f"    Loss: {loss2.item():.4f}")
    
    # Gradient 테스트
    print("\n[4] Gradient 테스트...")
    model.zero_grad()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"    Gradient norm: {grad_norm:.4f}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

