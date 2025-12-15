"""Dream visualization - generate and save dream sequences as GIFs."""

from pathlib import Path
from typing import List, Optional, Union

import imageio
import numpy as np
import torch

from ..models.world_model import WorldModel


class DreamGenerator:
    """Generates dream sequences from seed observations."""

    def __init__(
        self,
        world_model: WorldModel,
        device: str = "cuda",
    ):
        """
        Initialize dream generator.

        Args:
            world_model: Trained world model
            device: Device to run on
        """
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.model = world_model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_dream(
        self,
        seed_frames: torch.Tensor,
        actions: Union[List[int], int],
        num_frames: int = 50,
    ) -> np.ndarray:
        """
        Generate dream sequence from seed observation.

        Args:
            seed_frames: Seed observation (C, H, W) or (1, C, H, W)
            actions: List of action indices or single action to repeat
            num_frames: Number of frames to generate

        Returns:
            Generated frames as numpy array (num_frames+1, H, W)
        """
        # Prepare seed
        if seed_frames.dim() == 3:
            seed_frames = seed_frames.unsqueeze(0)
        seed_frames = seed_frames.to(self.device)

        # Prepare action sequence
        if isinstance(actions, int):
            action_seq = [actions] * num_frames
        elif len(actions) < num_frames:
            # Repeat last action if sequence is too short
            action_seq = list(actions) + [actions[-1]] * (num_frames - len(actions))
        else:
            action_seq = actions[:num_frames]

        # Generate dream
        frames = self.model.dream(seed_frames, action_seq)
        # frames: (num_frames+1, C, H, W)

        # Take last channel of each frame stack for visualization
        # This shows the most recent frame in each stack
        frames_np = frames[:, -1].cpu().numpy()  # (T, H, W)

        # Convert to uint8
        frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)

        return frames_np

    def save_gif(
        self,
        frames: np.ndarray,
        output_path: str,
        fps: int = 15,
    ) -> None:
        """
        Save frames as animated GIF.

        Args:
            frames: Frames array (T, H, W) or (T, H, W, C)
            output_path: Path to save GIF
            fps: Frames per second
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Ensure frames are in the right format
        if frames.ndim == 3:
            # Grayscale - convert to RGB for better compatibility
            frames_rgb = np.stack([frames] * 3, axis=-1)
        else:
            frames_rgb = frames

        # Save with imageio
        imageio.mimsave(
            output_path,
            frames_rgb,
            fps=fps,
            loop=0,  # Loop forever
        )
        print(f"Saved dream GIF to {output_path}")

    def dream_and_save(
        self,
        seed_frames: torch.Tensor,
        actions: Union[List[int], int],
        output_path: str,
        num_frames: int = 50,
        fps: int = 15,
    ) -> np.ndarray:
        """
        Generate dream and save as GIF.

        Args:
            seed_frames: Seed observation
            actions: Action sequence or single action
            output_path: Path to save GIF
            num_frames: Number of frames to generate
            fps: Frames per second

        Returns:
            Generated frames array
        """
        frames = self.generate_dream(seed_frames, actions, num_frames)
        self.save_gif(frames, output_path, fps)
        return frames

    @torch.no_grad()
    def dream_comparison(
        self,
        seed_frames: torch.Tensor,
        actions: List[List[int]],
        labels: Optional[List[str]] = None,
        num_frames: int = 50,
    ) -> np.ndarray:
        """
        Generate multiple dream sequences for comparison.

        Args:
            seed_frames: Seed observation (shared across all dreams)
            actions: List of action sequences
            labels: Optional labels for each sequence
            num_frames: Number of frames per dream

        Returns:
            Combined frames array (T, H, W*num_sequences)
        """
        all_frames = []

        for action_seq in actions:
            frames = self.generate_dream(seed_frames, action_seq, num_frames)
            all_frames.append(frames)

        # Concatenate horizontally
        combined = np.concatenate(all_frames, axis=2)  # (T, H, W*N)

        return combined

    def save_comparison_gif(
        self,
        seed_frames: torch.Tensor,
        actions: List[List[int]],
        output_path: str,
        labels: Optional[List[str]] = None,
        num_frames: int = 50,
        fps: int = 15,
    ) -> None:
        """
        Save comparison GIF with multiple action sequences.

        Args:
            seed_frames: Seed observation
            actions: List of action sequences
            output_path: Path to save GIF
            labels: Optional labels
            num_frames: Frames per dream
            fps: Frames per second
        """
        combined = self.dream_comparison(
            seed_frames, actions, labels, num_frames
        )
        self.save_gif(combined, output_path, fps)


def create_action_sequence(
    action: int,
    length: int,
    variation: Optional[List[tuple]] = None,
) -> List[int]:
    """
    Create an action sequence with optional variations.

    Args:
        action: Base action to repeat
        length: Sequence length
        variation: List of (start, end, action) tuples for variations

    Returns:
        Action sequence
    """
    seq = [action] * length

    if variation is not None:
        for start, end, var_action in variation:
            for i in range(start, min(end, length)):
                seq[i] = var_action

    return seq
