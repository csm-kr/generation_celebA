"""
Losses Module (JiT-style)

Flow Matching Loss와 Sampler를 제공합니다.
Reference: https://github.com/LTH14/JiT
"""

from .flow_matching_loss import FlowMatchingLoss, flow_matching_loss
from .sampler import FlowSampler, EulerSampler, euler_sample


def build_loss(loss_type: str = "flow_matching", **kwargs) -> FlowMatchingLoss:
    """
    Loss 함수를 생성합니다.
    
    Args:
        loss_type: Loss 타입 ('flow_matching')
        **kwargs: 추가 인자 (P_mean, P_std, noise_scale, t_eps)
        
    Returns:
        Loss 모듈
    """
    if loss_type.lower() == "flow_matching":
        return FlowMatchingLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: ['flow_matching']")


def build_sampler(sampler_type: str = "euler", **kwargs) -> FlowSampler:
    """
    Sampler를 생성합니다.
    
    Args:
        sampler_type: Sampler 타입 ('euler', 'heun')
        **kwargs: 추가 인자 (num_steps, noise_scale)
        
    Returns:
        Sampler 인스턴스
    """
    if sampler_type.lower() in ["euler", "heun"]:
        return FlowSampler(method=sampler_type.lower(), **kwargs)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}. Available: ['euler', 'heun']")


__all__ = [
    "FlowMatchingLoss",
    "flow_matching_loss",
    "FlowSampler",
    "EulerSampler",  # Backward compatibility
    "euler_sample",
    "build_loss",
    "build_sampler",
]

