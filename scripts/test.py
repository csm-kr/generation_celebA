"""
Test/Sampling Script for CelebGen

Generate samples from trained model.

Usage:
    python scripts/test.py --exp_name my_exp --checkpoint best.pt --num_samples 16
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import imageio

import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import build_model
from src.losses import EulerSampler


def get_args():
    parser = argparse.ArgumentParser(description="CelebGen Sampling")
    
    # Experiment
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Checkpoint filename")
    
    # Sampling
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples")
    parser.add_argument("--num_steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--save_gif", action="store_true", help="Save trajectory GIF")
    parser.add_argument("--save_grid", action="store_true", default=True, help="Save grid image")
    parser.add_argument("--save_individual", action="store_true", help="Save individual images")
    
    # Model (for loading)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--model_size", type=str, default="base")
    
    args = parser.parse_args()
    return args


def denormalize(x):
    """[-1, 1] -> [0, 1]"""
    return (x + 1) / 2


def create_gif(trajectory, save_path, duration=0.1):
    """Create GIF from trajectory."""
    frames = []
    for x in trajectory:
        img = x[0]
        img = denormalize(img).clamp(0, 1)
        img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames.append(img)
    imageio.mimsave(save_path, frames, duration=duration)


def main():
    args = get_args()
    
    print("=" * 60)
    print("CelebGen Sampling")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Setup paths
    exp_dir = Path("experiments") / args.exp_name
    checkpoint_path = exp_dir / "checkpoints" / args.checkpoint
    
    if args.output_dir is None:
        output_dir = exp_dir / "test_samples"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint if available
    if "args" in checkpoint:
        saved_args = checkpoint["args"]
        args.image_size = saved_args.get("image_size", args.image_size)
        args.patch_size = saved_args.get("patch_size", args.patch_size)
        args.model_size = saved_args.get("model_size", args.model_size)
    
    # Build model
    print(f"\nBuilding model: DiT-{args.model_size}")
    model = build_model(
        model_type="dit",
        img_size=args.image_size,
        patch_size=args.patch_size,
        model_size=args.model_size,
    ).to(device)
    
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
    
    # Sampler
    sampler = EulerSampler(num_steps=args.num_steps)
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples with {args.num_steps} steps...")
    
    torch.manual_seed(args.seed)
    
    samples, trajectory = sampler.sample(
        model=model,
        shape=(args.num_samples, 3, args.image_size, args.image_size),
        device=device,
        return_trajectory=True,
        progress=True,
    )
    
    print(f"Samples shape: {samples.shape}")
    print(f"Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    # Save outputs
    print(f"\nSaving to: {output_dir}")
    
    # Grid image
    if args.save_grid:
        nrow = int(np.ceil(np.sqrt(args.num_samples)))
        grid = make_grid(denormalize(samples), nrow=nrow, padding=2, normalize=False)
        grid_path = output_dir / f"grid_seed{args.seed}.png"
        save_image(grid, grid_path)
        print(f"  Grid saved: {grid_path}")
    
    # GIF
    if args.save_gif:
        gif_path = output_dir / f"trajectory_seed{args.seed}.gif"
        create_gif(trajectory, gif_path, duration=0.05)
        print(f"  GIF saved: {gif_path}")
    
    # Individual images
    if args.save_individual:
        individual_dir = output_dir / f"individual_seed{args.seed}"
        individual_dir.mkdir(exist_ok=True)
        for i, img in enumerate(samples):
            img = denormalize(img).clamp(0, 1)
            save_image(img, individual_dir / f"{i:04d}.png")
        print(f"  Individual images saved: {individual_dir}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

