"""Combined world model integrating VAE and dynamics."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import VAE
from .dynamics import DynamicsModel


class WorldModel(nn.Module):
    """Combined VAE + Dynamics model for world modeling."""

    def __init__(self, vae: VAE, dynamics: DynamicsModel):
        """
        Initialize world model.

        Args:
            vae: Trained VAE model
            dynamics: Trained dynamics model
        """
        super().__init__()
        self.vae = vae
        self.dynamics = dynamics

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode observation to latent space.

        Args:
            x: Observation tensor (B, C, H, W)

        Returns:
            Latent vector (B, latent_dim)
        """
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to observation.

        Args:
            z: Latent vector (B, latent_dim)

        Returns:
            Decoded observation (B, C, H, W)
        """
        return self.vae.decode(z)

    def predict_next_latent(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next latent given current latent and action.

        Args:
            z: Current latent (B, latent_dim)
            action: Action (B,) integers or (B, num_actions) one-hot
            hidden: Dynamics hidden state

        Returns:
            Tuple of (z_next, hidden)
        """
        # Convert action to one-hot if needed
        if action.dim() == 1:
            action = F.one_hot(action, self.dynamics.num_actions).float()

        return self.dynamics(z, action, hidden)

    def predict_next(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next observation given current observation and action.

        Args:
            x: Current observation (B, C, H, W)
            action: Action (B,) integers or (B, num_actions) one-hot
            hidden: Dynamics hidden state

        Returns:
            Tuple of:
                - x_next_pred: Predicted next observation (B, C, H, W)
                - z_next: Predicted next latent (B, latent_dim)
                - hidden: Updated hidden state
        """
        # Encode current observation
        z = self.encode(x)

        # Predict next latent
        z_next, hidden = self.predict_next_latent(z, action, hidden)

        # Decode to observation
        x_next_pred = self.decode(z_next)

        return x_next_pred, z_next, hidden

    def dream(
        self,
        x_seed: torch.Tensor,
        actions: Union[List[int], torch.Tensor],
    ) -> torch.Tensor:
        """
        Generate dream sequence from seed observation.

        Args:
            x_seed: Seed observation (1, C, H, W) or (C, H, W)
            actions: Action sequence - list of ints or tensor (T,)

        Returns:
            Generated frames (T+1, C, H, W)
        """
        device = x_seed.device

        # Ensure batch dimension
        if x_seed.dim() == 3:
            x_seed = x_seed.unsqueeze(0)

        # Convert actions to tensor if needed
        if isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.long, device=device)

        T = len(actions)

        # Encode seed
        z = self.encode(x_seed)  # (1, latent_dim)

        # Initialize hidden state
        hidden = self.dynamics.init_hidden(1, device)

        # Generate dream sequence
        frames = [self.decode(z)]

        for t in range(T):
            action = actions[t:t+1]  # Keep as tensor
            action_onehot = F.one_hot(action, self.dynamics.num_actions).float()

            z, hidden = self.dynamics(z, action_onehot, hidden)
            frames.append(self.decode(z))

        # Stack frames
        frames = torch.cat(frames, dim=0)  # (T+1, C, H, W)

        return frames

    def dream_from_latent(
        self,
        z_seed: torch.Tensor,
        actions: Union[List[int], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate dream sequence from seed latent.

        Args:
            z_seed: Seed latent (1, latent_dim) or (latent_dim,)
            actions: Action sequence

        Returns:
            Tuple of:
                - frames: Generated frames (T+1, C, H, W)
                - latents: Latent sequence (T+1, latent_dim)
        """
        device = z_seed.device

        # Ensure batch dimension
        if z_seed.dim() == 1:
            z_seed = z_seed.unsqueeze(0)

        # Convert actions to tensor if needed
        if isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.long, device=device)

        # Add batch dimension to actions for rollout
        actions = actions.unsqueeze(0)  # (1, T)

        # Rollout dynamics
        z_sequence, _ = self.dynamics.rollout(z_seed, actions)  # (1, T+1, latent_dim)
        z_sequence = z_sequence.squeeze(0)  # (T+1, latent_dim)

        # Decode all latents to frames
        frames = self.decode(z_sequence)  # (T+1, C, H, W)

        return frames, z_sequence

    @classmethod
    def from_checkpoints(
        cls,
        vae_checkpoint: str,
        dynamics_checkpoint: str,
        vae_config: dict,
        dynamics_config: dict,
        device: str = "cpu",
    ) -> "WorldModel":
        """
        Load world model from checkpoints.

        Args:
            vae_checkpoint: Path to VAE checkpoint
            dynamics_checkpoint: Path to dynamics checkpoint
            vae_config: VAE configuration dict
            dynamics_config: Dynamics configuration dict
            device: Device to load models to

        Returns:
            Loaded WorldModel
        """
        # Create VAE
        vae = VAE(
            in_channels=vae_config.get("in_channels", 4),
            latent_dim=vae_config.get("latent_dim", 64),
            beta=vae_config.get("beta", 0.0001),
        )

        # Load VAE weights
        vae_state = torch.load(vae_checkpoint, map_location=device)
        if "model_state_dict" in vae_state:
            vae.load_state_dict(vae_state["model_state_dict"])
        else:
            vae.load_state_dict(vae_state)

        # Create dynamics model
        dynamics = DynamicsModel(
            latent_dim=dynamics_config.get("latent_dim", 64),
            num_actions=dynamics_config.get("num_actions", 18),
            hidden_size=dynamics_config.get("hidden_size", 256),
            num_layers=dynamics_config.get("num_layers", 1),
        )

        # Load dynamics weights
        dyn_state = torch.load(dynamics_checkpoint, map_location=device)
        if "model_state_dict" in dyn_state:
            dynamics.load_state_dict(dyn_state["model_state_dict"])
        else:
            dynamics.load_state_dict(dyn_state)

        # Create world model
        model = cls(vae, dynamics)
        model.to(device)
        model.eval()

        return model
