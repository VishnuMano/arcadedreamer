"""Interactive pygame visualization for real-time model predictions."""

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass

from ..data.preprocessing import FrameStacker
from ..models.world_model import WorldModel


class InteractiveVisualizer:
    """Side-by-side visualization of real game vs model predictions."""

    # Default key mappings for Atari games
    DEFAULT_KEY_MAPPING: Dict[int, int] = {
        pygame.K_SPACE: 1,   # Fire
        pygame.K_RIGHT: 2,   # Right
        pygame.K_LEFT: 3,    # Left
        pygame.K_UP: 4,      # Up
        pygame.K_DOWN: 5,    # Down
    }

    def __init__(
        self,
        world_model: WorldModel,
        game_name: str,
        device: str = "cuda",
        frame_size: int = 64,
        display_scale: int = 4,
        fps: int = 30,
        key_mapping: Optional[Dict[int, int]] = None,
    ):
        """
        Initialize interactive visualizer.

        Args:
            world_model: Trained world model
            game_name: Atari game name
            device: Device to run model on
            frame_size: Size of preprocessed frames
            display_scale: Scale factor for display
            fps: Target frames per second
            key_mapping: Custom key to action mapping
        """
        if not PYGAME_AVAILABLE:
            raise ImportError(
                "pygame is required for interactive mode. "
                "Install with: pip install pygame"
            )

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.model = world_model.to(self.device)
        self.model.eval()

        self.game_name = game_name
        self.frame_size = frame_size
        self.display_scale = display_scale
        self.display_size = frame_size * display_scale
        self.fps = fps
        self.key_mapping = key_mapping or self.DEFAULT_KEY_MAPPING

        # Create environment
        self.env = gym.make(game_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        # Frame stacker
        self.stacker = FrameStacker(stack_size=4, frame_size=frame_size)

        # Initialize pygame
        pygame.init()
        # Three panels: Real | Predicted | Overlay
        self.screen = pygame.display.set_mode(
            (self.display_size * 3, self.display_size)
        )
        pygame.display.set_caption(f"ArcadeDreamer - {game_name}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        # Colors for overlay
        self.real_color = (0, 255, 0)      # Green
        self.pred_color = (255, 0, 0)      # Red
        self.overlap_color = (255, 255, 0) # Yellow

    def _frame_to_surface(
        self,
        frame: np.ndarray,
    ) -> pygame.Surface:
        """
        Convert frame to pygame surface.

        Args:
            frame: Frame array (H, W) or (H, W, C)

        Returns:
            Scaled pygame surface
        """
        # Ensure uint8
        if frame.dtype in [np.float32, np.float64]:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

        # Handle grayscale
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        # Create surface (pygame expects (W, H, C) for make_surface)
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surface = pygame.transform.scale(
            surface, (self.display_size, self.display_size)
        )
        return surface

    def _create_overlay(
        self,
        real: np.ndarray,
        predicted: np.ndarray,
    ) -> np.ndarray:
        """
        Create overlay visualization.

        Green = real only, Red = predicted only, Yellow = overlap

        Args:
            real: Real frame (H, W)
            predicted: Predicted frame (H, W)

        Returns:
            RGB overlay (H, W, 3)
        """
        # Ensure uint8
        if real.dtype in [np.float32, np.float64]:
            real = (real * 255).clip(0, 255).astype(np.uint8)
        if predicted.dtype in [np.float32, np.float64]:
            predicted = (predicted * 255).clip(0, 255).astype(np.uint8)

        overlay = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)

        # Red channel = predicted
        overlay[:, :, 0] = predicted
        # Green channel = real
        overlay[:, :, 1] = real
        # Blue channel = overlap (minimum)
        overlay[:, :, 2] = np.minimum(real, predicted)

        return overlay

    def _get_action(self) -> int:
        """
        Get action from keyboard input.

        Returns:
            Action index (0 = NOOP if no key pressed)
        """
        keys = pygame.key.get_pressed()
        for key, action in self.key_mapping.items():
            if keys[key]:
                return action
        return 0  # NOOP

    def _draw_labels(self) -> None:
        """Draw panel labels on screen."""
        labels = ["Real", "Predicted", "Overlay"]
        for i, label in enumerate(labels):
            text = self.font.render(label, True, (255, 255, 255))
            # Add background for readability
            bg_rect = text.get_rect()
            bg_rect.topleft = (i * self.display_size + 5, 5)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect.inflate(4, 2))
            self.screen.blit(text, (i * self.display_size + 5, 5))

    def _draw_info(self, action: int, reward: float, episode: int) -> None:
        """Draw game info at bottom of screen."""
        info_text = f"Action: {action} | Reward: {reward:.1f} | Episode: {episode}"
        text = self.font.render(info_text, True, (255, 255, 255))
        bg_rect = text.get_rect()
        bg_rect.bottomleft = (5, self.display_size - 5)
        pygame.draw.rect(self.screen, (0, 0, 0), bg_rect.inflate(4, 2))
        self.screen.blit(text, (5, self.display_size - 25))

    @torch.no_grad()
    def run(self) -> None:
        """Run interactive visualization loop."""
        print(f"Starting interactive mode for {self.game_name}")
        print("Controls:")
        print("  Arrow keys: Move")
        print("  Space: Fire")
        print("  ESC: Quit")

        # Reset environment
        obs, _ = self.env.reset()
        self.stacker.reset()

        # Initialize dynamics hidden state
        hidden = None

        # Episode tracking
        episode = 1
        total_reward = 0.0

        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        # Reset on R key
                        obs, _ = self.env.reset()
                        self.stacker.reset()
                        hidden = None
                        episode += 1
                        total_reward = 0.0

            # Get action from keyboard
            action = self._get_action()

            # Step environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Get real preprocessed frame
            stacked = self.stacker.add_frame(obs)
            real_frame = stacked[-1]  # Most recent frame

            # Get model prediction
            if len(self.stacker.frames) >= 4:
                x = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
                action_tensor = torch.tensor([action], device=self.device)

                pred_obs, _, hidden = self.model.predict_next(
                    x, action_tensor, hidden
                )
                pred_frame = pred_obs[0, -1].cpu().numpy()  # Last channel
            else:
                pred_frame = real_frame

            # Create visualizations
            real_surface = self._frame_to_surface(real_frame)
            pred_surface = self._frame_to_surface(pred_frame)
            overlay = self._create_overlay(real_frame, pred_frame)
            overlay_surface = self._frame_to_surface(overlay)

            # Draw to screen
            self.screen.fill((0, 0, 0))
            self.screen.blit(real_surface, (0, 0))
            self.screen.blit(pred_surface, (self.display_size, 0))
            self.screen.blit(overlay_surface, (self.display_size * 2, 0))

            # Draw labels and info
            self._draw_labels()
            self._draw_info(action, total_reward, episode)

            pygame.display.flip()

            # Handle episode end
            if done:
                obs, _ = self.env.reset()
                self.stacker.reset()
                hidden = None
                episode += 1
                total_reward = 0.0
            else:
                obs = next_obs

            # Cap framerate
            self.clock.tick(self.fps)

        # Cleanup
        pygame.quit()
        self.env.close()
        print("Interactive mode ended.")


class SideBySideVisualizer(InteractiveVisualizer):
    """Simplified two-panel visualization (Real | Predicted)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Resize screen for two panels
        self.screen = pygame.display.set_mode(
            (self.display_size * 2, self.display_size)
        )

    def run(self) -> None:
        """Run simplified visualization loop."""
        print(f"Starting side-by-side mode for {self.game_name}")

        obs, _ = self.env.reset()
        self.stacker.reset()
        hidden = None
        episode = 1
        total_reward = 0.0

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            action = self._get_action()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

            stacked = self.stacker.add_frame(obs)
            real_frame = stacked[-1]

            if len(self.stacker.frames) >= 4:
                x = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
                action_tensor = torch.tensor([action], device=self.device)
                pred_obs, _, hidden = self.model.predict_next(
                    x, action_tensor, hidden
                )
                pred_frame = pred_obs[0, -1].cpu().numpy()
            else:
                pred_frame = real_frame

            real_surface = self._frame_to_surface(real_frame)
            pred_surface = self._frame_to_surface(pred_frame)

            self.screen.fill((0, 0, 0))
            self.screen.blit(real_surface, (0, 0))
            self.screen.blit(pred_surface, (self.display_size, 0))

            # Labels
            for i, label in enumerate(["Real", "Predicted"]):
                text = self.font.render(label, True, (255, 255, 255))
                self.screen.blit(text, (i * self.display_size + 5, 5))

            pygame.display.flip()

            if done:
                obs, _ = self.env.reset()
                self.stacker.reset()
                hidden = None
                episode += 1
                total_reward = 0.0
            else:
                obs = next_obs

            self.clock.tick(self.fps)

        pygame.quit()
        self.env.close()
