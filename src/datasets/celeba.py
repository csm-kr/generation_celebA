"""
CelebA Dataset for Rectified Flow Training

CelebA 데이터셋을 로드하고 전처리하는 클래스입니다.
이미지를 [-1, 1] 범위로 정규화하여 Flow Matching 학습에 적합하게 변환합니다.
"""

import os
from typing import Optional, Callable, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
from PIL import Image


class CelebADataset(Dataset):
    """
    CelebA Dataset wrapper for Rectified Flow training.
    
    Args:
        root: 데이터셋 루트 디렉토리
        split: 'train', 'valid', 'test' 중 선택
        image_size: 출력 이미지 크기 (정사각형)
        download: 데이터셋 자동 다운로드 여부
        transform: 추가 변환 (기본 변환 후 적용)
    """
    
    def __init__(
        self,
        root: str = "./data",
        split: str = "train",
        image_size: int = 64,
        download: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        
        # 기본 변환: CelebA 이미지를 중앙 크롭 후 리사이즈, [-1, 1] 정규화
        self.base_transform = transforms.Compose([
            transforms.CenterCrop(178),  # CelebA 이미지 중앙 크롭 (얼굴 영역)
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
        ])
        
        self.additional_transform = transform
        
        # CelebA 데이터셋 로드
        self.dataset = CelebA(
            root=root,
            split=split,
            target_type="attr",
            transform=None,  # 변환은 __getitem__에서 수행
            download=download,
        )
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            image: 정규화된 이미지 텐서 [C, H, W], 범위 [-1, 1]
        """
        image, _ = self.dataset[idx]  # attr은 사용하지 않음
        
        # 기본 변환 적용
        image = self.base_transform(image)
        
        # 추가 변환 적용 (있는 경우)
        if self.additional_transform is not None:
            image = self.additional_transform(image)
            
        return image
    
    @staticmethod
    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        """
        [-1, 1] 범위의 텐서를 [0, 1] 범위로 변환합니다.
        
        Args:
            tensor: 정규화된 텐서
            
        Returns:
            [0, 1] 범위의 텐서
        """
        return (tensor + 1) / 2
    
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """
        텐서를 PIL 이미지로 변환합니다.
        
        Args:
            tensor: [C, H, W] 또는 [B, C, H, W] 형태의 텐서
            
        Returns:
            PIL Image
        """
        if tensor.dim() == 4:
            tensor = tensor[0]  # 첫 번째 이미지만 사용
            
        # [-1, 1] -> [0, 1] -> [0, 255]
        tensor = CelebADataset.denormalize(tensor)
        tensor = tensor.clamp(0, 1)
        tensor = (tensor * 255).byte()
        
        # [C, H, W] -> [H, W, C]
        numpy_image = tensor.permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(numpy_image)


def get_celeba_dataloader(
    root: str = "./data",
    split: str = "train",
    image_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    download: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    CelebA DataLoader를 생성합니다.
    
    Args:
        root: 데이터셋 루트 디렉토리
        split: 'train', 'valid', 'test'
        image_size: 이미지 크기
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
        shuffle: 셔플 여부
        download: 다운로드 여부
        pin_memory: GPU 전송 최적화
        
    Returns:
        DataLoader 인스턴스
    """
    dataset = CelebADataset(
        root=root,
        split=split,
        image_size=image_size,
        download=download,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # 학습 안정성을 위해 마지막 불완전한 배치 제거
    )
    
    return dataloader


if __name__ == "__main__":
    """
    CelebA Dataset 테스트: 시각화 및 DataLoader shape 확인
    """
    import cv2
    import numpy as np
    
    # 1. Dataset 생성 및 10개 이미지 시각화
    print("Dataset 생성 중...")
    dataset = CelebADataset(root="./data", split="train", image_size=64)
    print(f"총 이미지 수: {len(dataset):,}")
    
    print("\n10개 이미지 시각화 (아무 키나 누르면 다음)")
    for i in range(10):
        img = dataset[i]  # [C, H, W], [-1, 1]
        img = CelebADataset.denormalize(img)  # [0, 1]
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # [H, W, C]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"CelebA Sample {i}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 2. DataLoader shape 확인
    print("\nDataLoader 테스트...")
    dataloader = get_celeba_dataloader(root="./data", batch_size=32, num_workers=0)
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")  # [32, 3, 64, 64]

