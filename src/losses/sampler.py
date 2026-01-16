"""
Sampler for Rectified Flow (JiT-style)

Reference: https://github.com/LTH14/JiT

Supports:
- Euler method
- Heun method (2nd order)
- noise_scale parameter
"""

import torch
import torch.nn as nn
from typing import Optional, List, Callable
from tqdm import tqdm


class FlowSampler:
    """
    JiT-style sampler for Rectified Flow.
    
    t=0 (noise) 에서 t=1 (clean image)로 ODE를 적분합니다.
    
    Args:
        num_steps: Number of sampling steps
        noise_scale: Initial noise scale (default: 1.0)
        t_eps: Numerical stability epsilon
        method: 'euler' or 'heun'
    """
    
    def __init__(
        self,
        num_steps: int = 50,
        noise_scale: float = 1.0,
        t_eps: float = 1e-5,
        method: str = "euler",
    ):
        self.num_steps = num_steps
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.method = method
        
    def _get_velocity(
        self, 
        model: nn.Module, 
        z: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute velocity from model prediction."""
        x_pred = model(z, t)
        t_expand = t.view(-1, 1, 1, 1)
        v_pred = (x_pred - z) / (1 - t_expand).clamp(min=self.t_eps)
        return v_pred
    
    def _euler_step(
        self,
        model: nn.Module,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        """Single Euler step."""
        v_pred = self._get_velocity(model, z, t)
        dt = t_next - t
        return z + dt.view(-1, 1, 1, 1) * v_pred
    
    def _heun_step(
        self,
        model: nn.Module,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        """Single Heun step (2nd order)."""
        dt = t_next - t
        dt_expand = dt.view(-1, 1, 1, 1)
        
        # First evaluation
        v_pred_t = self._get_velocity(model, z, t)
        z_euler = z + dt_expand * v_pred_t
        
        # Second evaluation
        v_pred_next = self._get_velocity(model, z_euler, t_next)
        
        # Average
        v_avg = 0.5 * (v_pred_t + v_pred_next)
        return z + dt_expand * v_avg
        
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        device: torch.device = None,
        seed: Optional[int] = None,
        return_trajectory: bool = False,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples.
        
        Args:
            model: DiT model (predicts x_pred)
            shape: Output shape (B, C, H, W)
            device: Device to use
            seed: Random seed for reproducibility
            return_trajectory: Whether to return full trajectory
            progress: Show progress bar
            
        Returns:
            samples: Generated images [B, C, H, W]
            trajectory: (optional) List of intermediate states
        """
        if device is None:
            device = next(model.parameters()).device
            
        if seed is not None:
            torch.manual_seed(seed)
            
        B = shape[0]
        
        # Start from scaled noise at t=0
        z = torch.randn(shape, device=device) * self.noise_scale
        
        # Time steps: 0 -> 1
        timesteps = torch.linspace(0.0, 1.0, self.num_steps + 1, device=device)
        
        trajectory = [z.clone()] if return_trajectory else None
        
        # Select step function
        step_fn = self._heun_step if self.method == "heun" else self._euler_step
        
        iterator = range(self.num_steps)
        if progress:
            iterator = tqdm(iterator, desc=f"Sampling ({self.method})")
            
        for i in iterator:
            t = timesteps[i]
            t_next = timesteps[i + 1]
            
            # Batch timesteps
            t_batch = torch.full((B,), t.item(), device=device)
            t_next_batch = torch.full((B,), t_next.item(), device=device)
            
            # Use Euler for last step (as in JiT)
            if i == self.num_steps - 1:
                z = self._euler_step(model, z, t_batch, t_next_batch)
            else:
                z = step_fn(model, z, t_batch, t_next_batch)
            
            if return_trajectory:
                trajectory.append(z.clone())
                
        # Clamp to valid range
        z = z.clamp(-1, 1)
        
        if return_trajectory:
            return z, trajectory
        return z


# Backward compatibility alias
EulerSampler = FlowSampler


def euler_sample(
    model: nn.Module,
    num_samples: int = 1,
    image_size: int = 64,
    num_steps: int = 50,
    noise_scale: float = 1.0,
    device: torch.device = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Functional interface for Euler sampling.
    
    Args:
        model: DiT model
        num_samples: Number of samples to generate
        image_size: Image size
        num_steps: Number of sampling steps
        noise_scale: Initial noise scale
        device: Device to use
        seed: Random seed
        
    Returns:
        samples: Generated images [N, 3, H, W]
    """
    sampler = FlowSampler(num_steps=num_steps, noise_scale=noise_scale, method="euler")
    shape = (num_samples, 3, image_size, image_size)
    return sampler.sample(model, shape, device=device, seed=seed)


if __name__ == "__main__":
    """
    FlowSampler 테스트 (JiT-style)
    """
    import sys
    sys.path.insert(0, '/workspace/CelebGen')
    from src.models import build_model
    
    print("=" * 60)
    print("FlowSampler Test (JiT-style)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 모델 생성 (작은 모델로 테스트)
    print("\n[1] 모델 생성...")
    model = build_model(model_size="small", depth=6).to(device)
    model.eval()
    
    # Euler Sampler 테스트
    print("\n[2] Euler Sampler 테스트...")
    sampler_euler = FlowSampler(num_steps=10, noise_scale=1.0, method="euler")
    
    samples = sampler_euler.sample(
        model=model,
        shape=(4, 3, 64, 64),
        device=device,
        seed=42,
    )
    print(f"    Samples shape: {samples.shape}")
    print(f"    Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    # Heun Sampler 테스트
    print("\n[3] Heun Sampler 테스트...")
    sampler_heun = FlowSampler(num_steps=10, noise_scale=1.0, method="heun")
    
    samples_heun = sampler_heun.sample(
        model=model,
        shape=(4, 3, 64, 64),
        device=device,
        seed=42,
    )
    print(f"    Samples shape: {samples_heun.shape}")
    print(f"    Samples range: [{samples_heun.min():.3f}, {samples_heun.max():.3f}]")
    
    # Trajectory 테스트
    print("\n[4] Trajectory 테스트...")
    samples, trajectory = sampler_euler.sample(
        model=model,
        shape=(2, 3, 64, 64),
        device=device,
        seed=42,
        return_trajectory=True,
        progress=False,
    )
    print(f"    Trajectory length: {len(trajectory)}")
    print(f"    First (noise): [{trajectory[0].min():.3f}, {trajectory[0].max():.3f}]")
    print(f"    Last (clean): [{trajectory[-1].min():.3f}, {trajectory[-1].max():.3f}]")
    
    # Functional interface 테스트
    print("\n[5] euler_sample 함수 테스트...")
    samples2 = euler_sample(
        model=model,
        num_samples=2,
        image_size=64,
        num_steps=10,
        noise_scale=1.0,
        device=device,
        seed=123,
    )
    print(f"    Samples shape: {samples2.shape}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

