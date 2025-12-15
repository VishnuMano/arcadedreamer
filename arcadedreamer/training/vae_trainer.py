"""VAE training loop."""

import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..models.vae import VAE
from .utils import save_checkpoint, AverageMeter


class VAETrainer:
    """Trainer for the VAE model."""

    def __init__(
        self,
        model: VAE,
        train_dataset,
        val_dataset=None,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 128,
        num_epochs: int = 50,
        num_workers: int = 4,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_every: int = 10,
        device: str = "cuda",
        val_split: float = 0.1,
    ):
        """
        Initialize VAE trainer.

        Args:
            model: VAE model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional, will split from train if None)
            lr: Learning rate
            weight_decay: Weight decay for Adam
            batch_size: Batch size
            num_epochs: Number of training epochs
            num_workers: Number of data loader workers
            checkpoint_dir: Directory to save checkpoints
            checkpoint_every: Save checkpoint every N epochs
            device: Device to train on
            val_split: Fraction of training data to use for validation if val_dataset is None
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every

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
            model.parameters(),
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
            "train_recon": [],
            "train_kl": [],
            "val_loss": [],
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        loss_meter = AverageMeter()
        recon_meter = AverageMeter()
        kl_meter = AverageMeter()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs}",
            leave=True,
        )

        for batch in pbar:
            x = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            x_recon, mu, logvar, z = self.model(x)

            # Compute loss
            loss, recon_loss, kl_loss = self.model.loss_function(
                x, x_recon, mu, logvar
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update meters
            batch_size = x.size(0)
            loss_meter.update(loss.item(), batch_size)
            recon_meter.update(recon_loss.item(), batch_size)
            kl_meter.update(kl_loss.item(), batch_size)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "recon": f"{recon_meter.avg:.4f}",
                "kl": f"{kl_meter.avg:.4f}",
            })

        # Step scheduler
        self.scheduler.step()

        return {
            "loss": loss_meter.avg,
            "recon_loss": recon_meter.avg,
            "kl_loss": kl_meter.avg,
        }

    @torch.no_grad()
    def validate(self) -> Optional[float]:
        """
        Run validation.

        Returns:
            Validation loss or None if no validation set
        """
        if self.val_loader is None:
            return None

        self.model.eval()
        loss_meter = AverageMeter()

        for batch in self.val_loader:
            x = batch.to(self.device)

            x_recon, mu, logvar, z = self.model(x)
            loss, _, _ = self.model.loss_function(x, x_recon, mu, logvar)

            loss_meter.update(loss.item(), x.size(0))

        return loss_meter.avg

    def save(self, epoch: int, metrics: Dict) -> None:
        """
        Save checkpoint.

        Args:
            epoch: Current epoch
            metrics: Training metrics
        """
        path = self.checkpoint_dir / f"vae_epoch_{epoch + 1}.pt"
        save_checkpoint(
            str(path),
            self.model,
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

        print(f"Training VAE for {self.num_epochs} epochs on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_recon"].append(train_metrics["recon_loss"])
            self.history["train_kl"].append(train_metrics["kl_loss"])

            # Validate
            val_loss = self.validate()
            if val_loss is not None:
                self.history["val_loss"].append(val_loss)

            # Print epoch summary
            lr = self.scheduler.get_last_lr()[0]
            summary = (
                f"Epoch {epoch + 1}/{self.num_epochs}: "
                f"loss={train_metrics['loss']:.4f}, "
                f"recon={train_metrics['recon_loss']:.4f}, "
                f"kl={train_metrics['kl_loss']:.4f}, "
                f"lr={lr:.6f}"
            )
            if val_loss is not None:
                summary += f", val_loss={val_loss:.4f}"
            print(summary)

            # Save best model
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(epoch, {"val_loss": val_loss, **train_metrics})
                print(f"  New best validation loss: {val_loss:.4f}")

            # Periodic checkpoint
            if (epoch + 1) % self.checkpoint_every == 0:
                self.save(epoch, train_metrics)

        # Save final checkpoint
        self.save(self.num_epochs - 1, train_metrics)

        print("Training complete!")
        return self.history
