"""Data collection from Atari games using Gymnasium and ale-py."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from tqdm import tqdm

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass

from .preprocessing import preprocess_frame, FrameStacker


# Game ID mapping for multi-game training
GAME_IDS: Dict[str, int] = {
    "BreakoutNoFrameskip-v4": 0,
    "PongNoFrameskip-v4": 1,
    "SpaceInvadersNoFrameskip-v4": 2,
}


def get_game_id(game_name: str) -> int:
    """Get numeric ID for a game name."""
    if game_name in GAME_IDS:
        return GAME_IDS[game_name]
    # Assign new ID for unknown games
    return hash(game_name) % 1000


class AtariDataCollector:
    """Collects training data from Atari games using random policy."""

    def __init__(
        self,
        game_name: str,
        save_dir: str,
        frame_size: int = 64,
        stack_size: int = 4,
    ):
        """
        Initialize data collector.

        Args:
            game_name: Name of the Atari game (e.g., 'BreakoutNoFrameskip-v4')
            save_dir: Directory to save collected data
            frame_size: Size to resize frames to
            stack_size: Number of frames to stack
        """
        self.game_name = game_name
        self.save_dir = Path(save_dir)
        self.frame_size = frame_size
        self.stack_size = stack_size
        self.game_id = get_game_id(game_name)

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize environment
        self.env = gym.make(game_name, render_mode=None)
        self.num_actions = self.env.action_space.n

    def collect(
        self,
        num_frames: int,
        seed: Optional[int] = None,
    ) -> str:
        """
        Collect training data using random policy.

        Args:
            num_frames: Number of frames to collect
            seed: Random seed for reproducibility

        Returns:
            Path to saved data file
        """
        if seed is not None:
            np.random.seed(seed)

        # Storage for transitions
        observations = []
        actions = []
        next_observations = []
        rewards = []
        dones = []
        game_ids = []

        # Frame stacker for current and next observations
        stacker = FrameStacker(self.stack_size, self.frame_size)

        # Reset environment
        obs, info = self.env.reset(seed=seed)
        stacker.reset()

        # Add initial frame to stacker
        current_stacked = stacker.add_frame(obs)

        print(f"Collecting {num_frames} frames from {self.game_name}...")
        pbar = tqdm(total=num_frames, desc="Collecting")

        frame_count = 0
        episode_count = 0

        while frame_count < num_frames:
            # Random action
            action = self.env.action_space.sample()

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Get next stacked frames
            next_stacked = stacker.add_frame(next_obs)

            # Store transition (only if we have full stacks)
            if len(stacker.frames) >= self.stack_size:
                observations.append(current_stacked.copy())
                actions.append(action)
                next_observations.append(next_stacked.copy())
                rewards.append(reward)
                dones.append(done)
                game_ids.append(self.game_id)

                frame_count += 1
                pbar.update(1)

            current_stacked = next_stacked

            if done:
                # Reset environment and stacker
                obs, info = self.env.reset()
                stacker.reset()
                current_stacked = stacker.add_frame(obs)
                episode_count += 1

        pbar.close()
        self.env.close()

        print(f"Collected {frame_count} frames over {episode_count} episodes")

        # Convert to numpy arrays
        data = {
            "observations": np.array(observations, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "next_observations": np.array(next_observations, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=bool),
            "game_ids": np.array(game_ids, dtype=np.int64),
        }

        # Save data
        save_path = self.save_dir / f"{self.game_name}.npz"
        np.savez_compressed(save_path, **data)
        print(f"Saved data to {save_path}")

        return str(save_path)

    def get_seed_frames(
        self,
        num_frames: int = 4,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get seed frames for dreaming.

        Args:
            num_frames: Number of frames to collect
            seed: Random seed

        Returns:
            Stacked frames (stack_size, frame_size, frame_size)
        """
        env = gym.make(self.game_name, render_mode=None)
        stacker = FrameStacker(self.stack_size, self.frame_size)

        obs, _ = env.reset(seed=seed)
        stacker.add_frame(obs)

        # Take random actions to get diverse starting state
        for _ in range(num_frames + np.random.randint(10, 50)):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            stacker.add_frame(obs)
            if terminated or truncated:
                obs, _ = env.reset()
                stacker.reset()
                stacker.add_frame(obs)

        env.close()
        return stacker.get_stacked()


def collect_all_games(
    games: List[str],
    frames_per_game: int,
    save_dir: str,
    frame_size: int = 64,
    stack_size: int = 4,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Collect data from multiple games.

    Args:
        games: List of game names
        frames_per_game: Number of frames to collect per game
        save_dir: Directory to save data
        frame_size: Frame size
        stack_size: Stack size
        seed: Random seed

    Returns:
        List of paths to saved data files
    """
    paths = []
    for game in games:
        collector = AtariDataCollector(
            game_name=game,
            save_dir=save_dir,
            frame_size=frame_size,
            stack_size=stack_size,
        )
        path = collector.collect(frames_per_game, seed=seed)
        paths.append(path)

    return paths
