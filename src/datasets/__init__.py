"""
Datasets Module

데이터셋 빌더 함수를 제공합니다.
"""

from typing import Optional
from torch.utils.data import DataLoader

from .celeba import CelebADataset, get_celeba_dataloader


def build_dataset(
    name: str = "celeba",
    root: str = "./data",
    split: str = "train",
    image_size: int = 64,
    download: bool = True,
    **kwargs,
) -> CelebADataset:
    """
    데이터셋을 생성합니다.
    
    Args:
        name: 데이터셋 이름 ('celeba')
        root: 데이터셋 루트 디렉토리
        split: 'train', 'valid', 'test'
        image_size: 이미지 크기
        download: 다운로드 여부
        **kwargs: 추가 인자
        
    Returns:
        Dataset 인스턴스
    """
    if name.lower() == "celeba":
        return CelebADataset(
            root=root,
            split=split,
            image_size=image_size,
            download=download,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}. Available: ['celeba']")


def build_dataloader(
    name: str = "celeba",
    root: str = "./data",
    split: str = "train",
    image_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    download: bool = True,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """
    DataLoader를 생성합니다.
    
    Args:
        name: 데이터셋 이름 ('celeba')
        root: 데이터셋 루트 디렉토리
        split: 'train', 'valid', 'test'
        image_size: 이미지 크기
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
        shuffle: 셔플 여부 (None이면 train일 때 True)
        download: 다운로드 여부
        pin_memory: GPU 전송 최적화
        **kwargs: 추가 인자
        
    Returns:
        DataLoader 인스턴스
    """
    # shuffle 기본값 설정
    if shuffle is None:
        shuffle = (split == "train")
    
    if name.lower() == "celeba":
        return get_celeba_dataloader(
            root=root,
            split=split,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            download=download,
            pin_memory=pin_memory,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}. Available: ['celeba']")


__all__ = [
    "CelebADataset",
    "get_celeba_dataloader",
    "build_dataset",
    "build_dataloader",
]

