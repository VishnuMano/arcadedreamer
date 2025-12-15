"""Frame preprocessing utilities for Atari games."""

import cv2
import numpy as np
from typing import List, Optional


def preprocess_frame(
    frame: np.ndarray,
    target_size: int = 64,
) -> np.ndarray:
    """
    Convert RGB frame to grayscale and resize.

    Args:
        frame: (H, W, 3) RGB frame from ALE environment
        target_size: Output size (default 64x64)

    Returns:
        (target_size, target_size) grayscale frame, normalized to [0, 1]
    """
    # Convert to grayscale
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")

    # Resize to target size using INTER_AREA for downsampling
    resized = cv2.resize(
        gray,
        (target_size, target_size),
        interpolation=cv2.INTER_AREA,
    )

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized


def stack_frames(
    frames: List[np.ndarray],
    stack_size: int = 4,
) -> np.ndarray:
    """
    Stack consecutive frames into a single array.

    Args:
        frames: List of preprocessed frames (each H x W)
        stack_size: Number of frames to stack (default 4)

    Returns:
        (stack_size, H, W) stacked frames
    """
    if len(frames) == 0:
        raise ValueError("Cannot stack empty frame list")

    # Get frame shape from first frame
    frame_shape = frames[0].shape

    if len(frames) < stack_size:
        # Pad with zeros if not enough frames
        padding = [np.zeros(frame_shape, dtype=np.float32)] * (stack_size - len(frames))
        frames = padding + list(frames)

    # Take the last stack_size frames
    return np.stack(frames[-stack_size:], axis=0).astype(np.float32)


class FrameStacker:
    """Maintains a buffer of frames for stacking."""

    def __init__(self, stack_size: int = 4, frame_size: int = 64):
        """
        Initialize frame stacker.

        Args:
            stack_size: Number of frames to stack
            frame_size: Size of each frame (frame_size x frame_size)
        """
        self.stack_size = stack_size
        self.frame_size = frame_size
        self.frames: List[np.ndarray] = []

    def reset(self) -> None:
        """Clear the frame buffer."""
        self.frames = []

    def add_frame(self, frame: np.ndarray, preprocess: bool = True) -> np.ndarray:
        """
        Add a frame to the buffer and return the stacked frames.

        Args:
            frame: Raw frame from environment or preprocessed frame
            preprocess: Whether to preprocess the frame

        Returns:
            Stacked frames (stack_size, frame_size, frame_size)
        """
        if preprocess:
            processed = preprocess_frame(frame, self.frame_size)
        else:
            processed = frame

        self.frames.append(processed)

        # Keep only the last stack_size frames
        if len(self.frames) > self.stack_size:
            self.frames.pop(0)

        return stack_frames(self.frames, self.stack_size)

    def get_stacked(self) -> Optional[np.ndarray]:
        """
        Get current stacked frames without adding a new frame.

        Returns:
            Stacked frames or None if buffer is empty
        """
        if len(self.frames) == 0:
            return None
        return stack_frames(self.frames, self.stack_size)
