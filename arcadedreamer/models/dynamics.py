"""GRU-based dynamics model for latent space prediction."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    """
    GRU-based dynamics model for predicting next latent state.

    Architecture:
        Input: latent (64) + one-hot action (18) = 82
        Linear(82→256) + ReLU → (B, 256)
        GRU(input=256, hidden=256) → (B, 256), hidden: (1, B, 256)
        Linear(256→256) + ReLU → Linear(256→64) → z_next: (B, 64)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        num_actions: int = 18,
        hidden_size: int = 256,
        num_layers: int = 1,
    ):
        """
        Initialize dynamics model.

        Args:
            latent_dim: Dimension of latent space
            num_actions: Number of possible actions (ALE maximum is 18)
            hidden_size: GRU hidden state size
            num_layers: Number of GRU layers
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input size is latent + one-hot action
        input_size = latent_dim + num_actions

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
        )

        # GRU for temporal dynamics
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output projection to predict next latent
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, latent_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step prediction.

        Args:
            z: Current latent state (B, latent_dim)
            action: One-hot encoded action (B, num_actions)
            hidden: GRU hidden state (num_layers, B, hidden_size)

        Returns:
            Tuple of:
                - z_next: Predicted next latent (B, latent_dim)
                - hidden: Updated hidden state (num_layers, B, hidden_size)
        """
        # Concatenate latent and action
        x = torch.cat([z, action], dim=-1)  # (B, latent_dim + num_actions)

        # Project to hidden size
        x = self.input_proj(x)  # (B, hidden_size)

        # Add sequence dimension for GRU
        x = x.unsqueeze(1)  # (B, 1, hidden_size)

        # GRU forward
        output, hidden = self.gru(x, hidden)  # output: (B, 1, hidden_size)

        # Remove sequence dimension
        output = output.squeeze(1)  # (B, hidden_size)

        # Project to latent space
        z_next = self.output_proj(output)  # (B, latent_dim)

        return z_next, hidden

    def rollout(
        self,
        z_start: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-step rollout for dreaming.

        Args:
            z_start: Starting latent state (B, latent_dim)
            actions: Sequence of actions (B, T) integers or (B, T, num_actions) one-hot
            hidden: Initial hidden state

        Returns:
            Tuple of:
                - z_sequence: Predicted latent states (B, T+1, latent_dim)
                - hidden: Final hidden state
        """
        B = z_start.size(0)
        device = z_start.device

        # Determine sequence length
        if actions.dim() == 2:
            T = actions.size(1)
        else:
            T = actions.size(1)

        # Initialize hidden if needed
        if hidden is None:
            hidden = self.init_hidden(B, device)

        # Convert integer actions to one-hot if needed
        if actions.dim() == 2 and actions.dtype in [torch.int32, torch.int64, torch.long]:
            actions_onehot = F.one_hot(actions, self.num_actions).float()
        else:
            actions_onehot = actions

        # Collect latent sequence
        z_sequence = [z_start]
        z = z_start

        for t in range(T):
            action_t = actions_onehot[:, t]  # (B, num_actions)
            z, hidden = self.forward(z, action_t, hidden)
            z_sequence.append(z)

        # Stack sequence
        z_sequence = torch.stack(z_sequence, dim=1)  # (B, T+1, latent_dim)

        return z_sequence, hidden

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Initialize hidden state.

        Args:
            batch_size: Batch size
            device: Device

        Returns:
            Initial hidden state (num_layers, batch_size, hidden_size)
        """
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
        )

    def predict_sequence(
        self,
        z_sequence: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        use_teacher_forcing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next latents for a sequence (for training).

        Args:
            z_sequence: Sequence of latent states (B, T, latent_dim)
            actions: Sequence of actions (B, T) or (B, T, num_actions)
            hidden: Initial hidden state
            use_teacher_forcing: Whether to use ground truth latents

        Returns:
            Tuple of:
                - z_pred: Predicted next latents (B, T, latent_dim)
                - hidden: Final hidden state
        """
        B, T, _ = z_sequence.shape
        device = z_sequence.device

        if hidden is None:
            hidden = self.init_hidden(B, device)

        # Convert actions to one-hot if needed
        if actions.dim() == 2 and actions.dtype in [torch.int32, torch.int64, torch.long]:
            actions_onehot = F.one_hot(actions, self.num_actions).float()
        else:
            actions_onehot = actions

        z_predictions = []
        z = z_sequence[:, 0]  # Start from first latent

        for t in range(T):
            action_t = actions_onehot[:, t]
            z_next, hidden = self.forward(z, action_t, hidden)
            z_predictions.append(z_next)

            # Teacher forcing: use ground truth for next step
            if use_teacher_forcing and t < T - 1:
                z = z_sequence[:, t + 1]
            else:
                z = z_next

        z_pred = torch.stack(z_predictions, dim=1)  # (B, T, latent_dim)

        return z_pred, hidden
