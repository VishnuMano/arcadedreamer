"""Dynamics model training loop with scheduled sampling."""

import random
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..models.vae import VAE, Encoder
from ..models.dynamics import DynamicsModel
from .utils import save_checkpoint, AverageMeter


class DynamicsTrainer:
    """Trainer for the dynamics model with scheduled sampling."""

    def __init__(
        self,
        dynamics_model: DynamicsModel,
        vae_encoder: Encoder,
        train_dataset,
        val_dataset=None,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 128,
        num_epochs: int = 100,
        num_workers: int = 4,
        scheduled_sampling_prob: float = 0.5,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_every: int = 10,
        device: str = "cuda",
        val_split: float = 0.1,
    ):
        """
        Initialize dynamics trainer.

        Args:
            dynamics_model: Dynamics model to train
            vae_encoder: Frozen VAE encoder for latent encoding
            train_dataset: Training dataset (DynamicsDataset)
            val_dataset: Validation dataset (optional)
            lr: Learning rate
            weight_decay: Weight decay for Adam
            batch_size: Batch size
            num_epochs: Number of training epochs
            num_workers: Number of data loader workers
            scheduled_sampling_prob: Probability of using ground truth during training
            checkpoint_dir: Directory to save checkpoints
            checkpoint_every: Save checkpoint every N epochs
            device: Device to train on
            val_split: Fraction for validation if val_dataset is None
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dynamics = dynamics_model.to(self.device)
        self.encoder = vae_encoder.to(self.device)
        self.num_epochs = num_epochs
        self.scheduled_sampling_prob = scheduled_sampling_prob
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every

        # Freeze encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Split dataset if no validation set provided
        if val_dataset is None and val_split > 0:
            val_size = int(len(train_dataset) * val_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size]
            )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

        # Optimizer and scheduler
        self.optimizer = Adam(
            dynamics_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

    @torch.no_grad()
    def encode_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of frames to latent space.

        Args:
            frames: Frame batch (B, T, C, H, W) or (B*T, C, H, W)

        Returns:
            Latent vectors (B, T, latent_dim) or (B*T, latent_dim)
        """
        original_shape = frames.shape
        has_time_dim = len(original_shape) == 5

        if has_time_dim:
            B, T, C, H, W = original_shape
            frames = frames.view(B * T, C, H, W)

        # Encode using mean (deterministic)
        mu, _ = self.encoder(frames)

        if has_time_dim:
            mu = mu.view(B, T, -1)

        return mu

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with scheduled sampling.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.dynamics.train()
        loss_meter = AverageMeter()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs}",
            leave=True,
        )

        for batch in pbar:
            frames, actions, next_frames = batch
            frames = frames.to(self.device)
            actions = actions.to(self.device)
            next_frames = next_frames.to(self.device)

            B, T = frames.shape[:2]

            self.optimizer.zero_grad()

            # Encode all frames to latent space
            with torch.no_grad():
                z_current = self.encode_batch(frames)  # (B, T, latent_dim)
                z_target = self.encode_batch(next_frames)  # (B, T, latent_dim)

            # Convert actions to one-hot
            actions_onehot = F.one_hot(
                actions, self.dynamics.num_actions
            ).float()  # (B, T, num_actions)

            # Multi-step rollout with scheduled sampling
            hidden = self.dynamics.init_hidden(B, self.device)
            z_pred = z_current[:, 0]  # Start from first frame's latent

            total_loss = 0.0

            for t in range(T):
                # Scheduled sampling: decide whether to use ground truth
                if t > 0 and random.random() < self.scheduled_sampling_prob:
                    z_input = z_current[:, t]  # Use ground truth
                else:
                    z_input = z_pred  # Use prediction

                action_t = actions_onehot[:, t]  # (B, num_actions)
                z_pred, hidden = self.dynamics(z_input, action_t, hidden)

                # MSE loss between predicted and actual next latent
                step_loss = F.mse_loss(z_pred, z_target[:, t])
                total_loss += step_loss

            # Average loss over sequence
            loss = total_loss / T

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update meter
            loss_meter.update(loss.item(), B)

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss_meter.avg:.6f}"})

        # Step scheduler
        self.scheduler.step()

        return {"loss": loss_meter.avg}

    @torch.no_grad()
    def validate(self) -> Optional[float]:
        """
        Run validation.

        Returns:
            Validation loss or None if no validation set
        """
        if self.val_loader is None:
            return None

        self.dynamics.eval()
        loss_meter = AverageMeter()

        for batch in self.val_loader:
            frames, actions, next_frames = batch
            frames = frames.to(self.device)
            actions = actions.to(self.device)
            next_frames = next_frames.to(self.device)

            B, T = frames.shape[:2]

            # Encode frames
            z_current = self.encode_batch(frames)
            z_target = self.encode_batch(next_frames)

            # Convert actions to one-hot
            actions_onehot = F.one_hot(
                actions, self.dynamics.num_actions
            ).float()

            # Rollout without teacher forcing for validation
            hidden = self.dynamics.init_hidden(B, self.device)
            z_pred = z_current[:, 0]

            total_loss = 0.0

            for t in range(T):
                action_t = actions_onehot[:, t]
                z_pred, hidden = self.dynamics(z_pred, action_t, hidden)
                step_loss = F.mse_loss(z_pred, z_target[:, t])
                total_loss += step_loss

            loss = total_loss / T
            loss_meter.update(loss.item(), B)

        return loss_meter.avg

    def save(self, epoch: int, metrics: Dict) -> None:
        """
        Save checkpoint.

        Args:
            epoch: Current epoch
            metrics: Training metrics
        """
        path = self.checkpoint_dir / f"dynamics_epoch_{epoch + 1}.pt"
        save_checkpoint(
            str(path),
            self.dynamics,
            self.optimizer,
            self.scheduler,
            epoch,
            metrics,
        )

    def train(self) -> Dict:
        """
        Run full training loop.

        Returns:
            Training history
        """
        best_val_loss = float("inf")

        print(f"Training Dynamics Model for {self.num_epochs} epochs on {self.device}")
        print(f"Scheduled sampling probability: {self.scheduled_sampling_prob}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history["train_loss"].append(train_metrics["loss"])

            # Validate
            val_loss = self.validate()
            if val_loss is not None:
                self.history["val_loss"].append(val_loss)

            # Print epoch summary
            lr = self.scheduler.get_last_lr()[0]
            summary = (
                f"Epoch {epoch + 1}/{self.num_epochs}: "
                f"loss={train_metrics['loss']:.6f}, "
                f"lr={lr:.6f}"
            )
            if val_loss is not None:
                summary += f", val_loss={val_loss:.6f}"
            print(summary)

            # Save best model
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(epoch, {"val_loss": val_loss, **train_metrics})
                print(f"  New best validation loss: {val_loss:.6f}")

            # Periodic checkpoint
            if (epoch + 1) % self.checkpoint_every == 0:
                self.save(epoch, train_metrics)

        # Save final checkpoint
        self.save(self.num_epochs - 1, train_metrics)

        print("Training complete!")
        return self.history
