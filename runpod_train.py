#!/usr/bin/env python3
"""
RunPod training script for ArcadeDreamer.

Usage:
    1. Upload your collected data to RunPod volume or collect on GPU
    2. Run: python runpod_train.py --stage vae
    3. Run: python runpod_train.py --stage dynamics
    4. Download checkpoints when done
"""

import argparse
import os
import time
from pathlib import Path

import torch


def check_gpu():
    """Check GPU availability and print info."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"Memory: {gpu_mem:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")

        # H100 optimizations
        if "H100" in gpu_name:
            print("H100 detected - enabling TF32 for faster training")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        return "cuda"
    else:
        print("WARNING: No GPU detected, training will be slow!")
        return "cpu"


def train_vae(args):
    """Train VAE with GPU optimizations."""
    from arcadedreamer.models.vae import VAE
    from arcadedreamer.data.dataset import VAEDataset
    from arcadedreamer.training.vae_trainer import VAETrainer

    device = check_gpu()

    # Larger batch size for H100 (80GB memory)
    batch_size = args.batch_size or (512 if "H100" in torch.cuda.get_device_name(0) else 256)

    print(f"\nTraining VAE")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Data dir: {args.data_dir}")

    # Create dataset
    dataset = VAEDataset(
        data_dir=args.data_dir,
        games=None,  # Load all available games
        stack_size=4,
    )
    print(f"Dataset size: {len(dataset)} samples")

    # Create model
    model = VAE(
        in_channels=4,
        latent_dim=64,
        beta=0.0001,
    )

    # Use torch.compile for faster training on H100
    if hasattr(torch, 'compile') and device == "cuda":
        print("Using torch.compile for optimization")
        model = torch.compile(model)

    # Create trainer with optimized settings
    trainer = VAETrainer(
        model=model,
        train_dataset=dataset,
        lr=args.lr,
        weight_decay=1e-5,
        batch_size=batch_size,
        num_epochs=args.epochs,
        num_workers=8,  # More workers for faster data loading
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.save_every,
        device=device,
    )

    # Train
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


def train_dynamics(args):
    """Train dynamics model with GPU optimizations."""
    from arcadedreamer.models.vae import VAE
    from arcadedreamer.models.dynamics import DynamicsModel
    from arcadedreamer.data.dataset import DynamicsDataset
    from arcadedreamer.training.dynamics_trainer import DynamicsTrainer

    device = check_gpu()

    # Larger batch size for H100
    batch_size = args.batch_size or (512 if "H100" in torch.cuda.get_device_name(0) else 256)

    print(f"\nTraining Dynamics Model")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {args.epochs}")

    # Load VAE checkpoint
    vae_checkpoint = args.vae_checkpoint
    if vae_checkpoint is None:
        checkpoint_dir = Path(args.checkpoint_dir)
        vae_checkpoints = sorted(checkpoint_dir.glob("vae_epoch_*.pt"))
        if not vae_checkpoints:
            raise FileNotFoundError("No VAE checkpoint found. Train VAE first.")
        vae_checkpoint = str(vae_checkpoints[-1])

    print(f"Loading VAE from: {vae_checkpoint}")

    # Create and load VAE
    vae = VAE(in_channels=4, latent_dim=64, beta=0.0001)
    checkpoint = torch.load(vae_checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    # Handle torch.compile() prefix in saved weights
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    vae.load_state_dict(state_dict)

    # Create dataset
    dataset = DynamicsDataset(
        data_dir=args.data_dir,
        games=None,
        stack_size=4,
        sequence_length=10,
    )
    print(f"Dataset size: {len(dataset)} sequences")

    # Create dynamics model
    dynamics = DynamicsModel(
        latent_dim=64,
        num_actions=18,
        hidden_size=256,
        num_layers=1,
    )

    # Use torch.compile for faster training
    if hasattr(torch, 'compile') and device == "cuda":
        print("Using torch.compile for optimization")
        dynamics = torch.compile(dynamics)

    # Create trainer
    trainer = DynamicsTrainer(
        dynamics_model=dynamics,
        vae_encoder=vae.encoder,
        train_dataset=dataset,
        lr=args.lr,
        weight_decay=1e-5,
        batch_size=batch_size,
        num_epochs=args.epochs,
        num_workers=8,
        scheduled_sampling_prob=0.5,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.save_every,
        device=device,
    )

    # Train
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="RunPod GPU Training for ArcadeDreamer")
    parser.add_argument("--stage", type=str, required=True, choices=["vae", "dynamics"],
                        help="Training stage")
    parser.add_argument("--data-dir", type=str, default="./data/collected",
                        help="Directory with collected data")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--vae-checkpoint", type=str, default=None,
                        help="VAE checkpoint path (for dynamics training)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (default: 50 for VAE, 100 for dynamics)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: auto-detect based on GPU)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")

    args = parser.parse_args()

    # Set default epochs
    if args.epochs is None:
        args.epochs = 50 if args.stage == "vae" else 100

    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if args.stage == "vae":
        train_vae(args)
    else:
        train_dynamics(args)


if __name__ == "__main__":
    main()
