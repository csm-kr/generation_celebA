"""
Models Module

DiT 모델 및 관련 컴포넌트를 제공합니다.
"""

from .dit import DiT, build_dit, TimestepEmbedder, VisionRotaryEmbedding
from .adain_zero import AdaINZero, AdaINZeroBlock, ModulatedLayerNorm


def build_model(
    model_type: str = "dit",
    img_size: int = 64,
    patch_size: int = 4,
    model_size: str = "base",
    **kwargs,
) -> DiT:
    """
    모델을 생성합니다.
    
    Args:
        model_type: 모델 타입 ('dit')
        img_size: 이미지 크기 (default: 64)
        patch_size: 패치 크기 (default: 4)
        model_size: 모델 크기 ('small', 'base', 'large')
        **kwargs: 추가 인자
        
    Returns:
        생성된 모델
    """
    if model_type.lower() == "dit":
        return build_dit(
            img_size=img_size,
            patch_size=patch_size,
            model_size=model_size,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: ['dit']")


__all__ = [
    # Models
    "DiT",
    "build_dit",
    "build_model",
    # Time Embedding
    "TimestepEmbedder",
    # Conditioning
    "AdaINZero",
    "AdaINZeroBlock",
    "ModulatedLayerNorm",
]
