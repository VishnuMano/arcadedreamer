"""Variational Autoencoder for frame encoding."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Convolutional encoder for VAE.

    Architecture:
        Input: (B, 4, 64, 64)
        Conv2d(4→32, k=4, s=2, p=1) + ReLU  → (B, 32, 32, 32)
        Conv2d(32→64, k=4, s=2, p=1) + ReLU → (B, 64, 16, 16)
        Conv2d(64→128, k=4, s=2, p=1) + ReLU → (B, 128, 8, 8)
        Conv2d(128→256, k=4, s=2, p=1) + ReLU → (B, 256, 4, 4)
        Flatten → Linear(4096→512) + ReLU → (B, 512)
        mu: Linear(512→64), logvar: Linear(512→64)
    """

    def __init__(self, in_channels: int = 4, latent_dim: int = 64):
        """
        Initialize encoder.

        Args:
            in_channels: Number of input channels (stacked frames)
            latent_dim: Dimension of latent space
        """
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # After convolutions: 256 * 4 * 4 = 4096
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
        )

        # Latent space projections
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input tensor (B, in_channels, 64, 64)

        Returns:
            Tuple of (mu, logvar), each (B, latent_dim)
        """
        # Convolutional encoding
        h = self.conv_layers(x)  # (B, 256, 4, 4)

        # Flatten and project
        h = h.view(h.size(0), -1)  # (B, 4096)
        h = self.fc(h)  # (B, 512)

        # Latent distribution parameters
        mu = self.fc_mu(h)  # (B, latent_dim)
        logvar = self.fc_logvar(h)  # (B, latent_dim)

        return mu, logvar


class Decoder(nn.Module):
    """
    Convolutional decoder for VAE.

    Architecture:
        Input: (B, 64)
        Linear(64→512) + ReLU → Linear(512→4096) + ReLU
        Reshape → (B, 256, 4, 4)
        ConvT(256→128, k=4, s=2, p=1) + ReLU → (B, 128, 8, 8)
        ConvT(128→64, k=4, s=2, p=1) + ReLU → (B, 64, 16, 16)
        ConvT(64→32, k=4, s=2, p=1) + ReLU → (B, 32, 32, 32)
        ConvT(32→4, k=4, s=2, p=1) + Sigmoid → (B, 4, 64, 64)
    """

    def __init__(self, latent_dim: int = 64, out_channels: int = 4):
        """
        Initialize decoder.

        Args:
            latent_dim: Dimension of latent space
            out_channels: Number of output channels (stacked frames)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.out_channels = out_channels

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # Deconvolutional layers
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.

        Args:
            z: Latent vector (B, latent_dim)

        Returns:
            Reconstructed image (B, out_channels, 64, 64)
        """
        # Project and reshape
        h = self.fc(z)  # (B, 4096)
        h = h.view(h.size(0), 256, 4, 4)  # (B, 256, 4, 4)

        # Deconvolutional decoding
        x_recon = self.deconv_layers(h)  # (B, out_channels, 64, 64)

        return x_recon


class VAE(nn.Module):
    """Complete Variational Autoencoder model."""

    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int = 64,
        beta: float = 0.0001,
    ):
        """
        Initialize VAE.

        Args:
            in_channels: Number of input channels (stacked frames)
            latent_dim: Dimension of latent space
            beta: Weight for KL divergence term
        """
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon

        Args:
            mu: Mean of latent distribution (B, latent_dim)
            logvar: Log variance of latent distribution (B, latent_dim)

        Returns:
            Sampled latent vector (B, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input tensor (B, in_channels, 64, 64)

        Returns:
            Tuple of (x_recon, mu, logvar, z):
                - x_recon: Reconstructed input (B, in_channels, 64, 64)
                - mu: Latent mean (B, latent_dim)
                - logvar: Latent log variance (B, latent_dim)
                - z: Sampled latent (B, latent_dim)
        """
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        return x_recon, mu, logvar, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space (use mean for deterministic encoding).

        Args:
            x: Input tensor (B, in_channels, 64, 64)

        Returns:
            Latent vector (B, latent_dim)
        """
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.

        Args:
            z: Latent vector (B, latent_dim)

        Returns:
            Decoded image (B, in_channels, 64, 64)
        """
        return self.decoder(z)

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss.

        Loss = Reconstruction Loss + beta * KL Divergence

        Args:
            x: Original input (B, in_channels, 64, 64)
            x_recon: Reconstructed input (B, in_channels, 64, 64)
            mu: Latent mean (B, latent_dim)
            logvar: Latent log variance (B, latent_dim)

        Returns:
            Tuple of (total_loss, recon_loss, kl_loss)
        """
        batch_size = x.size(0)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction="sum") / batch_size

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from the prior distribution.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            Generated samples (num_samples, in_channels, 64, 64)
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
