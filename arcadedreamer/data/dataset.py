"""PyTorch Dataset classes for training."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    """Dataset for VAE training - returns stacked frames."""

    def __init__(
        self,
        data_dir: str,
        games: Optional[List[str]] = None,
        stack_size: int = 4,
    ):
        """
        Initialize VAE dataset.

        Args:
            data_dir: Directory containing .npz data files
            games: List of game names to load (loads all if None)
            stack_size: Number of stacked frames (for validation)
        """
        self.data_dir = Path(data_dir)
        self.stack_size = stack_size

        # Load all game data
        self.observations: List[np.ndarray] = []
        self.game_indices: List[int] = []

        data_files = list(self.data_dir.glob("*.npz"))
        if games is not None:
            data_files = [
                f for f in data_files
                if f.stem in games or any(g in f.stem for g in games)
            ]

        if len(data_files) == 0:
            raise ValueError(f"No data files found in {data_dir}")

        print(f"Loading data from {len(data_files)} files...")
        for file_path in data_files:
            data = np.load(file_path)
            obs = data["observations"]
            game_ids = data["game_ids"]

            # Store observations and game indices
            start_idx = len(self.observations)
            self.observations.extend(obs)
            self.game_indices.extend([int(game_ids[0])] * len(obs))

            print(f"  Loaded {len(obs)} samples from {file_path.name}")

        # Convert to numpy arrays for efficient indexing
        self.observations = np.array(self.observations, dtype=np.float32)

        print(f"Total samples: {len(self.observations)}")

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sample.

        Args:
            idx: Sample index

        Returns:
            Stacked frames tensor (stack_size, H, W)
        """
        obs = self.observations[idx]
        return torch.from_numpy(obs)


class DynamicsDataset(Dataset):
    """Dataset for dynamics model training - returns sequences."""

    def __init__(
        self,
        data_dir: str,
        games: Optional[List[str]] = None,
        stack_size: int = 4,
        sequence_length: int = 10,
    ):
        """
        Initialize dynamics dataset.

        Args:
            data_dir: Directory containing .npz data files
            games: List of game names to load
            stack_size: Number of stacked frames
            sequence_length: Length of sequences for training
        """
        self.data_dir = Path(data_dir)
        self.stack_size = stack_size
        self.sequence_length = sequence_length

        # Load all game data
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.next_observations: List[np.ndarray] = []
        self.dones: List[bool] = []

        # Track episode boundaries for each game
        self.episode_boundaries: List[int] = []

        data_files = list(self.data_dir.glob("*.npz"))
        if games is not None:
            data_files = [
                f for f in data_files
                if f.stem in games or any(g in f.stem for g in games)
            ]

        if len(data_files) == 0:
            raise ValueError(f"No data files found in {data_dir}")

        print(f"Loading data from {len(data_files)} files...")
        for file_path in data_files:
            data = np.load(file_path)

            obs = data["observations"]
            acts = data["actions"]
            next_obs = data["next_observations"]
            dones = data["dones"]

            # Add data
            offset = len(self.observations)
            self.observations.extend(obs)
            self.actions.extend(acts)
            self.next_observations.extend(next_obs)
            self.dones.extend(dones)

            # Find episode boundaries (where done=True)
            for i, done in enumerate(dones):
                if done:
                    self.episode_boundaries.append(offset + i)

            print(f"  Loaded {len(obs)} samples from {file_path.name}")

        # Convert to numpy arrays
        self.observations = np.array(self.observations, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.int64)
        self.next_observations = np.array(self.next_observations, dtype=np.float32)
        self.dones = np.array(self.dones, dtype=bool)
        self.episode_boundaries = set(self.episode_boundaries)

        # Build valid sequence indices (sequences that don't cross episode boundaries)
        self.valid_indices = self._build_valid_indices()

        print(f"Total samples: {len(self.observations)}")
        print(f"Valid sequences: {len(self.valid_indices)}")

    def _build_valid_indices(self) -> List[int]:
        """Build list of valid starting indices for sequences."""
        valid = []
        n = len(self.observations)

        for i in range(n - self.sequence_length):
            # Check if sequence crosses episode boundary
            crosses_boundary = False
            for j in range(i, i + self.sequence_length - 1):
                if j in self.episode_boundaries:
                    crosses_boundary = True
                    break

            if not crosses_boundary:
                valid.append(i)

        return valid

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sequence.

        Args:
            idx: Sequence index

        Returns:
            Tuple of:
                - frames: (seq_len, stack_size, H, W)
                - actions: (seq_len,)
                - next_frames: (seq_len, stack_size, H, W)
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length

        frames = self.observations[start_idx:end_idx]
        actions = self.actions[start_idx:end_idx]
        next_frames = self.next_observations[start_idx:end_idx]

        return (
            torch.from_numpy(frames),
            torch.from_numpy(actions),
            torch.from_numpy(next_frames),
        )


class CombinedDataset(Dataset):
    """Combined dataset from multiple games with balanced sampling."""

    def __init__(
        self,
        data_dir: str,
        games: List[str],
        stack_size: int = 4,
        mode: str = "vae",
        sequence_length: int = 10,
    ):
        """
        Initialize combined dataset.

        Args:
            data_dir: Directory containing data
            games: List of game names
            stack_size: Number of stacked frames
            mode: 'vae' or 'dynamics'
            sequence_length: Sequence length for dynamics mode
        """
        if mode == "vae":
            self.dataset = VAEDataset(data_dir, games, stack_size)
        else:
            self.dataset = DynamicsDataset(
                data_dir, games, stack_size, sequence_length
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]
