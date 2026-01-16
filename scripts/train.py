"""
Training Script for CelebGen

Multi-GPU training with DDP (spawn), TensorBoard logging, FID evaluation, GIF generation.

Usage:
    # With config file
    python scripts/train.py --config configs/default.yaml
    
    # With command line args
    python scripts/train.py --exp_name my_exp --epochs 50 --batch_size 32
    
    # Override config with command line
    python scripts/train.py --config configs/base_2gpu.yaml --lr 1e-5
    
Multi-GPU:
    python scripts/train.py --config configs/base_2gpu.yaml
"""

import os
import sys
import configargparse
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
import imageio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import build_dataset, build_dataloader
from src.models import build_model
from src.losses import FlowMatchingLoss, FlowSampler


# ============================================================================
# Config
# ============================================================================

def get_args():
    parser = configargparse.ArgumentParser(
        description="CelebGen Training",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument("--config", is_config_file=True, help="Config file path (YAML)")
    
    # Experiment
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Data
    parser.add_argument("--data_root", type=str, default="./data", help="Dataset root")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # Model
    parser.add_argument("--model_size", type=str, default="base", choices=["small", "base", "large"])
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (JiT uses 5e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    
    # Loss parameters (JiT-style)
    parser.add_argument("--P_mean", type=float, default=-0.8, help="Logit-normal mean")
    parser.add_argument("--P_std", type=float, default=0.8, help="Logit-normal std")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="Noise scale")
    parser.add_argument("--t_eps", type=float, default=1e-5, help="Numerical stability epsilon")
    
    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--warmup_start_factor", type=float, default=0.1, help="Warmup start factor")
    
    # Sampling
    parser.add_argument("--num_sampling_steps", type=int, default=50, help="ODE sampling steps")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples for visualization")
    parser.add_argument("--sampling_method", type=str, default="euler", choices=["euler", "heun"])
    
    # FID
    parser.add_argument("--fid_num_samples", type=int, default=1000, help="Number of samples for FID")
    parser.add_argument("--fid_batch_size", type=int, default=64, help="Batch size for FID generation")
    parser.add_argument("--fid_interval", type=int, default=10, help="Compute FID every N epochs (0=disable)")
    
    # Multi-GPU
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--dist_backend", type=str, default="nccl", help="Distributed backend")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    return args


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed(rank, world_size, backend="nccl"):
    """Initialize distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process."""
    return rank == 0


def save_checkpoint(state, path):
    """Save checkpoint."""
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    
    start_epoch = checkpoint.get("epoch", 0) + 1
    
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        
    print(f"Checkpoint loaded: {path} (epoch {start_epoch - 1})")
    return start_epoch


def denormalize(x):
    """[-1, 1] -> [0, 1]"""
    return (x + 1) / 2


def create_gif(trajectory, save_path, duration=0.1):
    """
    Create GIF from sampling trajectory.
    
    Args:
        trajectory: List of tensors [B, C, H, W]
        save_path: Output GIF path
        duration: Duration per frame
    """
    frames = []
    for t_idx, x in enumerate(trajectory):
        # Take first image from batch
        img = x[0]  # [C, H, W]
        img = denormalize(img).clamp(0, 1)
        img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames.append(img)
    
    imageio.mimsave(save_path, frames, duration=duration)


# ============================================================================
# Training
# ============================================================================

def get_lr_with_warmup(optimizer, epoch, warmup_epochs, warmup_start_factor, base_lr):
    """Calculate LR with linear warmup (epoch-based, JiT-style)."""
    if epoch < warmup_epochs:
        # Linear warmup: start_factor -> 1.0
        factor = warmup_start_factor + (1.0 - warmup_start_factor) * (epoch / warmup_epochs)
        return base_lr * factor
    return None  # Use scheduler


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    epoch,
    args,
    writer=None,
    rank=0,
    global_step_start=0,
):
    """Train for one epoch (JiT-style with epoch-based warmup)."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    base_lr = args.lr
    
    # Epoch-based warmup (JiT-style)
    if epoch < args.warmup_epochs:
        warmup_lr = get_lr_with_warmup(
            optimizer, epoch, args.warmup_epochs, 
            args.warmup_start_factor, base_lr
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process(rank))
    
    for batch_idx, x in enumerate(pbar):
        x = x.cuda(rank)
        global_step = global_step_start + batch_idx
        
        # Forward
        loss, info = criterion(model, x)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # Update stats
        total_loss += loss.item()
        num_batches += 1
        
        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        
        if is_main_process(rank) and batch_idx % args.log_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}",
                "t": f"{info.get('t_mean', 0):.2f}"
            })
            
            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
                writer.add_scalar("train/t_mean", info.get("t_mean", 0), global_step)
    
    # Step cosine scheduler (epoch-based, after warmup)
    if scheduler is not None and epoch >= args.warmup_epochs:
        scheduler.step()
    
    avg_loss = total_loss / num_batches
    return avg_loss, global_step_start + len(dataloader)


@torch.no_grad()
def evaluate(
    model,
    sampler,
    epoch,
    args,
    writer=None,
    exp_dir=None,
    rank=0,
):
    """
    Evaluate model: generate samples, compute FID, create GIF.
    """
    model.eval()
    
    if not is_main_process(rank):
        return None
    
    print(f"\n[Eval] Epoch {epoch}")
    
    # Generate samples with trajectory
    print("  Generating samples...")
    samples, trajectory = sampler.sample(
        model=model,
        shape=(args.num_samples, 3, args.image_size, args.image_size),
        device=torch.device(f"cuda:{rank}"),
        seed=args.seed,
        return_trajectory=True,
        progress=True,
    )
    
    # Save samples to TensorBoard
    if writer is not None:
        grid = make_grid(denormalize(samples), nrow=4, normalize=False)
        writer.add_image("samples/generated", grid, epoch)
        
        # Add trajectory images (first, middle, last)
        for i, idx in enumerate([0, len(trajectory)//2, -1]):
            traj_grid = make_grid(denormalize(trajectory[idx][:4]), nrow=2, normalize=False)
            writer.add_image(f"trajectory/step_{idx}", traj_grid, epoch)
    
    # Save GIF
    if exp_dir is not None:
        gif_dir = exp_dir / "gifs"
        gif_dir.mkdir(exist_ok=True)
        gif_path = gif_dir / f"epoch_{epoch:03d}.gif"
        create_gif(trajectory, gif_path, duration=0.05)
        print(f"  GIF saved: {gif_path}")
    
    # Save sample images
    if exp_dir is not None:
        samples_dir = exp_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        grid = make_grid(denormalize(samples), nrow=4, normalize=False)
        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(grid_np).save(samples_dir / f"epoch_{epoch:03d}.png")
    
    # FID computation (optional - only every fid_interval epochs)
    fid_score = None
    should_compute_fid = (
        args.fid_interval > 0 and 
        (epoch + 1) % args.fid_interval == 0
    )
    
    if should_compute_fid:
        try:
            from pytorch_fid import fid_score as fid_module
            
            print(f"  Computing FID ({args.fid_num_samples} samples)...")
            
            # Generate samples for FID
            fid_samples_dir = exp_dir / "fid_samples"
            fid_samples_dir.mkdir(exist_ok=True)
            
            # Clear old samples
            for f in fid_samples_dir.glob("*.png"):
                f.unlink()
            
            num_generated = 0
            while num_generated < args.fid_num_samples:
                batch_samples = sampler.sample(
                    model=model,
                    shape=(args.fid_batch_size, 3, args.image_size, args.image_size),
                    device=torch.device(f"cuda:{rank}"),
                    progress=False,
                )
                
                for i, img in enumerate(batch_samples):
                    if num_generated >= args.fid_num_samples:
                        break
                    img = denormalize(img).clamp(0, 1)
                    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(img_np).save(fid_samples_dir / f"{num_generated:05d}.png")
                    num_generated += 1
            
            print(f"  FID samples generated: {num_generated}")
            
        except ImportError:
            print("  [Warning] pytorch-fid not installed, skipping FID computation")
        except Exception as e:
            print(f"  [Warning] FID computation failed: {e}")
    
    if writer is not None and fid_score is not None:
        writer.add_scalar("eval/fid", fid_score, epoch)
    
    return fid_score


# ============================================================================
# Main Worker
# ============================================================================

def train_worker(rank, world_size, args):
    """Main training worker for each GPU."""
    
    # Setup distributed
    if world_size > 1:
        setup_distributed(rank, world_size, args.dist_backend)
    
    # Set seed
    set_seed(args.seed + rank)
    
    # Experiment directory
    exp_dir = Path("experiments") / args.exp_name
    if is_main_process(rank):
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "samples").mkdir(exist_ok=True)
    
    # TensorBoard
    writer = None
    if is_main_process(rank):
        writer = SummaryWriter(log_dir=exp_dir / "logs")
        print(f"TensorBoard: tensorboard --logdir {exp_dir / 'logs'}")
    
    # =========== Dataset ===========
    if is_main_process(rank):
        print("\n[1] Loading dataset...")
    
    dataset = build_dataset(
        name="celeba",
        root=args.data_root,
        split="train",
        image_size=args.image_size,
        download=True,
    )
    
    if is_main_process(rank):
        print(f"    Dataset size: {len(dataset)}")
    
    # =========== DataLoader ===========
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    # =========== Model ===========
    if is_main_process(rank):
        print("\n[2] Building model...")
    
    model = build_model(
        model_type="dit",
        img_size=args.image_size,
        patch_size=args.patch_size,
        model_size=args.model_size,
    ).cuda(rank)
    
    if is_main_process(rank):
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"    Model: DiT-{args.model_size}")
        print(f"    Parameters: {params:.2f}M")
    
    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # =========== Loss (JiT-style) ===========
    criterion = FlowMatchingLoss(
        P_mean=args.P_mean,
        P_std=args.P_std,
        noise_scale=args.noise_scale,
        t_eps=args.t_eps,
    )
    
    if is_main_process(rank):
        print(f"    Loss: FlowMatchingLoss (P_mean={args.P_mean}, P_std={args.P_std})")
    
    # =========== Optimizer ===========
    if is_main_process(rank):
        print("\n[3] Setting up optimizer...")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # =========== Scheduler ===========
    # Cosine scheduler (epoch-based, after warmup)
    scheduler = None
    if args.scheduler == "cosine":
        # Schedule from warmup_epochs to end
        T_max = args.epochs - args.warmup_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(T_max, 1),
            eta_min=args.lr * 0.01,
        )
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.epochs // 3,
            gamma=0.5,
        )
    
    # Note: Warmup is epoch-based (JiT-style)
    # warmup_epochs=5, warmup_start_factor=0.1 -> LR: 0.1x -> 1.0x
    
    # =========== Sampler (JiT-style) ===========
    sampler_eval = FlowSampler(
        num_steps=args.num_sampling_steps, 
        noise_scale=args.noise_scale,
        method=args.sampling_method,
    )
    
    # =========== Resume ===========
    start_epoch = 0
    if args.resume is not None:
        model_to_load = model.module if world_size > 1 else model
        start_epoch = load_checkpoint(args.resume, model_to_load, optimizer, scheduler)
    
    # =========== Training Loop ===========
    if is_main_process(rank):
        print("\n[4] Starting training...")
        print(f"    Epochs: {args.epochs}")
        print(f"    Batch size: {args.batch_size} x {world_size} GPUs = {args.batch_size * world_size}")
        print(f"    Learning rate: {args.lr}")
        print()
    
    best_fid = float("inf")
    global_step = start_epoch * len(dataloader)  # Track global iteration
    
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if world_size > 1:
            dataloader.sampler.set_epoch(epoch)
        
        # Train
        avg_loss, global_step = train_one_epoch(
            model=model.module if world_size > 1 else model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            args=args,
            writer=writer,
            rank=rank,
            global_step_start=global_step,
        )
        
        if is_main_process(rank):
            print(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}")
        
        # Eval (every epoch)
        model_eval = model.module if world_size > 1 else model
        fid_score = evaluate(
            model=model_eval,
            sampler=sampler_eval,
            epoch=epoch,
            args=args,
            writer=writer,
            exp_dir=exp_dir,
            rank=rank,
        )
        
        # Save checkpoint
        if is_main_process(rank):
            # Save latest
            checkpoint = {
                "epoch": epoch,
                "model": model_eval.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "args": vars(args),
            }
            save_checkpoint(checkpoint, exp_dir / "checkpoints" / "latest.pt")
            
            # Save periodic
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(checkpoint, exp_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
            
            # Save best (if FID available)
            if fid_score is not None and fid_score < best_fid:
                best_fid = fid_score
                save_checkpoint(checkpoint, exp_dir / "checkpoints" / "best.pt")
                print(f"  New best FID: {best_fid:.2f}")
    
    # Cleanup
    if writer is not None:
        writer.close()
    
    if world_size > 1:
        cleanup_distributed()
    
    if is_main_process(rank):
        print("\nTraining completed!")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    args = get_args()
    
    print("=" * 60)
    print("CelebGen Training")
    print("=" * 60)
    print(f"Experiment: {args.exp_name}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    if args.num_gpus > 1:
        # Multi-GPU training with spawn
        print(f"\nLaunching {args.num_gpus} GPU workers...")
        mp.spawn(
            train_worker,
            args=(args.num_gpus, args),
            nprocs=args.num_gpus,
            join=True,
        )
    else:
        # Single GPU training
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()

