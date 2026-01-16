# CelebGen

CelebA 데이터셋 기반 얼굴 생성 AI - **Rectified Flow** with **DiT (Diffusion Transformer)**

## 결과 (Results)

### Generation Process (Epoch 40)

![Generation GIF](experiments/celeba_base_2gpu/gifs/epoch_040.gif)

*t=0 (noise) → t=1 (generated face) trajectory*

---

## 아키텍처

```
64x64 Noisy Image
       ↓
   ┌─────────────────────────┐
   │      Patchify           │  4x4 patches → 256 tokens
   │      (Linear Embed)     │
   └─────────────────────────┘
       ↓
   ┌─────────────────────────┐
   │   Transformer Block     │  ← AdaIN-Zero (time conditioning)
   │   + RoPE Attention      │  ← RoPE (time → Q, K rotation)
   └─────────────────────────┘
       ↓ (× N layers)
   ┌─────────────────────────┐
   │    Linear Predict       │
   │    (Unpatchify)         │
   └─────────────────────────┘
       ↓
64x64 x-pred (velocity)
```

## 설치

```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```
CelebGen/
├── src/
│   ├── models/          # DiT 모델
│   ├── datasets/        # CelebA 데이터셋
│   └── losses/          # Flow Matching Loss
├── scripts/
│   ├── train.py         # 학습 스크립트
│   └── test.py          # 테스트/샘플링
├── eval/                # FID 평가
├── experiments/         # 실험 결과 저장
├── requirements.txt
└── README.md
```

---

## Datasets

### CelebA Dataset

CelebA 데이터셋을 로드하고 전처리하는 모듈입니다.

#### 파일 구조

```
src/datasets/
├── __init__.py      # build_dataset(), build_dataloader()
└── celeba.py        # CelebADataset 클래스
```

#### 사용법

```python
from src.datasets import build_dataset, build_dataloader

# Dataset 생성
dataset = build_dataset(
    name="celeba",
    root="./data",
    split="train",      # 'train', 'valid', 'test'
    image_size=64,
    download=True,
)

print(f"Total images: {len(dataset)}")  # 162,770

# 단일 이미지 로드
img = dataset[0]  # [3, 64, 64], range [-1, 1]

# DataLoader 생성
dataloader = build_dataloader(
    name="celeba",
    root="./data",
    split="train",
    image_size=64,
    batch_size=32,
    num_workers=4,
)

for batch in dataloader:
    # batch: [B, 3, 64, 64]
    break
```

#### 데이터 전처리

| 단계 | 설명 |
|------|------|
| Center Crop | 178x178 (얼굴 영역) |
| Resize | 64x64 |
| Normalize | `[-1, 1]` (mean=0.5, std=0.5) |

#### 유틸리티 함수

```python
from src.datasets import CelebADataset

# [-1, 1] → [0, 1]
img_01 = CelebADataset.denormalize(img)

# Tensor → PIL Image
pil_img = CelebADataset.tensor_to_pil(img)
```

---

## Models

### DiT (Diffusion Transformer)

64x64 이미지를 입력받아 velocity (x-pred)를 예측하는 Transformer 모델입니다.

#### 전체 모델 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                         DiT Model                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: x [B, 3, 64, 64]     t [B]                             │
│              │                 │                                │
│              ▼                 ▼                                │
│  ┌─────────────────┐   ┌─────────────────┐                     │
│  │   PatchEmbed    │   │  TimeEmbedding  │                     │
│  │  (Conv2d 4x4)   │   │  (RoPE + MLP)   │                     │
│  │ 64→16, 256 tok  │   │    → [B, 512]   │                     │
│  └────────┬────────┘   └────────┬────────┘                     │
│           │                     │                               │
│           ▼                     │                               │
│  ┌─────────────────┐            │                               │
│  │  + pos_embed    │            │                               │
│  │  [1, 256, 512]  │            │                               │
│  └────────┬────────┘            │                               │
│           │                     │                               │
│           ▼                     ▼                               │
│  ┌─────────────────────────────────────────┐                   │
│  │           DiT Block × 12 (base)          │                   │
│  │  ┌─────────────────────────────────────┐ │                   │
│  │  │  AdaIN-Zero ← time_emb              │ │                   │
│  │  │  Multi-Head Attention (8 heads)     │ │                   │
│  │  │  + RoPE (Q, K rotation)             │ │                   │
│  │  │  + Residual                         │ │                   │
│  │  ├─────────────────────────────────────┤ │                   │
│  │  │  AdaIN-Zero ← time_emb              │ │                   │
│  │  │  Feed Forward (512 → 2048 → 512)    │ │                   │
│  │  │  + Residual                         │ │                   │
│  │  └─────────────────────────────────────┘ │                   │
│  └─────────────────────────────────────────┘                   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │   LayerNorm     │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │   Unpatchify    │                                           │
│  │ (Linear → 4×4×3)│                                           │
│  │ 256 tok → 64×64 │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  Output: x_pred [B, 3, 64, 64]                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 파일 구조

```
src/models/
├── __init__.py      # build_model()
├── dit.py           # DiT 메인 모델
├── rope.py          # RoPE Time Embedding
└── adain_zero.py    # AdaIN-Zero Conditioning
```

#### 모델 크기

| Size | Embed Dim | Depth | Heads | MLP Dim | Parameters |
|------|-----------|-------|-------|---------|------------|
| small | 384 | 12 | 6 | 1536 | 29.69M |
| base | 512 | 12 | 8 | 2048 | 52.69M |
| large | 768 | 24 | 12 | 3072 | 231.73M |

#### Patch 구성 (64x64 이미지)

```
64×64 Image          Patch Size 4×4         256 Tokens
┌────────────────┐   ┌──┬──┬──┬──┐         ┌─────────────┐
│                │   │ 0│ 1│ 2│ 3│         │ Token 0     │
│                │   ├──┼──┼──┼──┤         │ Token 1     │
│    Image       │ → │ 4│ 5│ 6│ 7│    →    │ ...         │
│                │   ├──┼──┼──┼──┤         │ Token 255   │
│                │   │..│..│..│..│         └─────────────┘
└────────────────┘   └──┴──┴──┴──┘          [B, 256, D]
                      16×16 grid
```

#### 사용법

```python
from src.models import build_model, DiT

# 방법 1: build_model 사용
model = build_model(
    model_type="dit",
    img_size=64,
    patch_size=4,
    model_size="base",  # 'small', 'base', 'large'
)

# 방법 2: DiT 직접 생성
model = DiT(
    img_size=64,
    patch_size=4,
    in_channels=3,
    out_channels=3,
    embed_dim=512,
    depth=12,
    num_heads=8,
    mlp_ratio=4.0,
    dropout=0.0,
    use_rope=True,
)

# Forward
x = torch.randn(B, 3, 64, 64)  # noisy image
t = torch.rand(B)              # timestep [0, 1]
v = model(x, t)                # velocity prediction [B, 3, 64, 64]
```

#### 주요 컴포넌트

##### 1. RoPE (Rotary Position Embedding)

```python
from src.models import RoPE, TimeEmbedding

# RoPE for attention
rope = RoPE(dim=64)  # head_dim
cos, sin = rope(t)   # t: [B], cos/sin: [B, dim]

# TimeEmbedding (RoPE + MLP)
time_emb = TimeEmbedding(dim=512)
emb = time_emb(t)    # [B, 512]
```

##### 2. AdaIN-Zero (Adaptive Instance Normalization)

```python
from src.models import AdaINZero

adain = AdaINZero(dim=512, time_dim=512)

# x: [B, N, 512], time_emb: [B, 512]
out = adain(x, time_emb)

# 내부 연산:
# scale, shift = MLP(time_emb)  (zero-initialized)
# out = LayerNorm(x) * (1 + scale) + shift
```

#### DiT Block 구조

```
Input x
   ↓
┌──────────────────────┐
│  AdaIN-Zero (norm)   │ ← time_emb
└──────────────────────┘
   ↓
┌──────────────────────┐
│  Multi-Head Attention│ ← RoPE (Q, K rotation)
└──────────────────────┘
   ↓ (+residual)
┌──────────────────────┐
│  AdaIN-Zero (norm)   │ ← time_emb
└──────────────────────┘
   ↓
┌──────────────────────┐
│  Feed Forward (MLP)  │
└──────────────────────┘
   ↓ (+residual)
Output
```

---

## Training

### Config 파일로 학습 (권장)

```bash
# 기본 설정
python scripts/train.py --config configs/default.yaml

# Base 모델 + 2 GPU
python scripts/train.py --config configs/base_2gpu.yaml

# Large 모델 + 4 GPU
python scripts/train.py --config configs/large_4gpu.yaml

# Config 파일 + 일부 인자 override
python scripts/train.py --config configs/base_2gpu.yaml --lr 1e-5 --exp_name my_exp
```

### Config 파일 구조

```yaml
# configs/default.yaml
exp_name: celeba_default
seed: 42

# Data
data_root: ./data
image_size: 64
batch_size: 32

# Model
model_size: base  # small, base, large

# Training
epochs: 50
lr: 5e-6

# Scheduler
scheduler: cosine
warmup_iters: 1000
warmup_start_factor: 0.1

# Multi-GPU
num_gpus: 1
```

### 사용 가능한 Config 파일

| Config | 모델 | GPU | 배치 | LR |
|--------|------|-----|------|-----|
| `default.yaml` | base | 1 | 32 | 5e-6 |
| `base_2gpu.yaml` | base | 2 | 128 | 5e-6 |
| `large_4gpu.yaml` | large | 4 | 64 | 5e-6 |

### Command Line으로 학습

```bash
# 기본 학습
python scripts/train.py --exp_name my_experiment

# 커스텀 설정
python scripts/train.py \
    --exp_name celeba_custom \
    --model_size base \
    --batch_size 64 \
    --lr 5e-6 \
    --epochs 100
```

### Multi-GPU 학습 (DDP)

```bash
python scripts/train.py \
    --exp_name celeba_base_2gpu \
    --model_size base \
    --batch_size 128 \
    --num_gpus 2

# 2 GPU 학습
python scripts/train.py \
    --exp_name celeba_large_2gpu \
    --model_size large \
    --batch_size 256 \
    --num_gpus 2

# 4 GPU 학습 (배치 512 = 128 per GPU)
python scripts/train.py \
    --exp_name celeba_large_4gpu \
    --model_size large \
    --batch_size 128 \
    --num_gpus 4
```

### 주요 인자

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--exp_name` | 실험 이름 (필수) | - |
| `--model_size` | 모델 크기: small, base, large | base |
| `--batch_size` | GPU당 배치 크기 | 32 |
| `--epochs` | 학습 에폭 | 50 |
| `--lr` | Learning rate | 1e-4 |
| `--num_gpus` | 사용할 GPU 수 | 1 |
| `--resume` | 체크포인트 경로 (재개용) | None |
| `--num_sampling_steps` | 샘플링 스텝 수 | 50 |

### 권장 설정

| 설정 | GPU | batch_size | model_size | 예상 VRAM |
|------|-----|------------|------------|-----------|
| 테스트용 | 1 | 32 | small | ~8GB |
| 기본 | 1 | 64 | base | ~16GB |
| 고품질 | 2 | 256 | large | ~40GB x2 |
| 최대 | 4 | 128 | large | ~24GB x4 |

### Resume 학습

```bash
# 체크포인트에서 재개
python scripts/train.py \
    --exp_name celeba_large_b512 \
    --resume experiments/celeba_large_b512/checkpoints/latest.pt
```

---

## TensorBoard

학습 중 loss, 생성 이미지, trajectory 등을 실시간으로 모니터링할 수 있습니다.

### 실행 방법

```bash
# TensorBoard 서버 시작
tensorboard --logdir experiments/celeba_large_b512/logs --port 6006

# 브라우저에서 접속
# http://localhost:6006
```

### 로그 내용

| 탭 | 내용 |
|----|------|
| Scalars | train/loss, train/lr, eval/fid |
| Images | samples/generated, trajectory/step_* |

### 여러 실험 비교

```bash
# 모든 실험 로그 비교
tensorboard --logdir experiments/ --port 6006
```

---

## Sampling (Test)

학습된 모델로 이미지 생성

```bash
# 기본 샘플링
python scripts/test.py \
    --exp_name celeba_large_b512 \
    --checkpoint best.pt \
    --num_samples 16

# GIF 포함 샘플링
python scripts/test.py \
    --exp_name celeba_large_b512 \
    --checkpoint best.pt \
    --num_samples 16 \
    --save_gif \
    --num_steps 100
```

### 출력 파일

```
experiments/{exp_name}/
├── checkpoints/
│   ├── latest.pt      # 마지막 체크포인트
│   ├── best.pt        # 최고 FID 체크포인트
│   └── epoch_*.pt     # 에폭별 체크포인트
├── logs/              # TensorBoard 로그
├── samples/           # 에폭별 생성 이미지
├── gifs/              # Trajectory GIF
└── test_samples/      # 테스트 샘플
```

---

## 테스트

### Dataset 테스트

```bash
cd /workspace/CelebGen
python src/datasets/celeba.py
```

### Model 테스트

```bash
python -c "
from src.models import build_model
import torch

model = build_model(model_size='large')
x = torch.randn(2, 3, 64, 64).cuda()
t = torch.rand(2).cuda()
model = model.cuda()
v = model(x, t)
print(f'Input: {x.shape} → Output: {v.shape}')
"
```

### Loss & Sampler 테스트

```bash
python src/losses/flow_matching_loss.py
python src/losses/sampler.py
```

---

## License

MIT License

