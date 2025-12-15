#!/usr/bin/env python3
"""ArcadeDreamer - CLI for training and visualizing Atari world models."""

import argparse
import os
from pathlib import Path

import torch
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    # Look for config in package directory if not found at specified path
    if not os.path.exists(config_path):
        package_config = Path(__file__).parent / "arcadedreamer" / "configs" / "default.yaml"
        if package_config.exists():
            config_path = str(package_config)
        else:
            raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def cmd_collect(args):
    """Collect training data from Atari games."""
    from arcadedreamer.data.collector import AtariDataCollector, collect_all_games

    config = load_config(args.config)

    # Determine games to collect
    if args.game:
        games = [args.game]
    else:
        games = config["data"]["games"]

    # Determine number of frames
    frames = args.frames or config["data"]["frames_per_game"]

    # Data directory
    data_dir = config["data"]["data_dir"]

    print(f"Collecting data from {len(games)} game(s)")
    print(f"Frames per game: {frames}")
    print(f"Save directory: {data_dir}")

    # Collect from each game
    for game in games:
        print(f"\n{'='*50}")
        print(f"Collecting from {game}")
        print(f"{'='*50}")

        collector = AtariDataCollector(
            game_name=game,
            save_dir=data_dir,
            frame_size=config["data"]["frame_size"],
            stack_size=config["data"]["stack_size"],
        )
        collector.collect(num_frames=frames, seed=args.seed)

    print("\nData collection complete!")


def cmd_train(args):
    """Train VAE or dynamics model."""
    from arcadedreamer.models.vae import VAE
    from arcadedreamer.models.dynamics import DynamicsModel
    from arcadedreamer.data.dataset import VAEDataset, DynamicsDataset
    from arcadedreamer.training.vae_trainer import VAETrainer
    from arcadedreamer.training.dynamics_trainer import DynamicsTrainer

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training on device: {device}")

    if args.stage == "vae":
        print("\n" + "="*50)
        print("Training VAE")
        print("="*50)

        # Create dataset
        dataset = VAEDataset(
            data_dir=config["data"]["data_dir"],
            games=config["data"]["games"],
            stack_size=config["data"]["stack_size"],
        )

        # Create model
        model = VAE(
            in_channels=config["data"]["stack_size"],
            latent_dim=config["vae"]["latent_dim"],
            beta=config["vae"]["beta"],
        )

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create trainer
        trainer = VAETrainer(
            model=model,
            train_dataset=dataset,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            batch_size=config["training"]["batch_size"],
            num_epochs=config["training"]["vae_epochs"],
            num_workers=config["training"]["num_workers"],
            checkpoint_dir="./checkpoints",
            checkpoint_every=config["training"]["checkpoint_every"],
            device=device,
        )

        # Train
        trainer.train()

    elif args.stage == "dynamics":
        print("\n" + "="*50)
        print("Training Dynamics Model")
        print("="*50)

        # Load trained VAE
        vae = VAE(
            in_channels=config["data"]["stack_size"],
            latent_dim=config["vae"]["latent_dim"],
            beta=config["vae"]["beta"],
        )

        vae_checkpoint = args.vae_checkpoint
        if vae_checkpoint is None:
            # Find latest VAE checkpoint
            checkpoint_dir = Path("./checkpoints")
            vae_checkpoints = sorted(checkpoint_dir.glob("vae_epoch_*.pt"))
            if not vae_checkpoints:
                raise FileNotFoundError("No VAE checkpoint found. Train VAE first.")
            vae_checkpoint = str(vae_checkpoints[-1])

        print(f"Loading VAE from {vae_checkpoint}")
        checkpoint = torch.load(vae_checkpoint, map_location=device)
        vae.load_state_dict(checkpoint["model_state_dict"])

        # Create dataset
        dataset = DynamicsDataset(
            data_dir=config["data"]["data_dir"],
            games=config["data"]["games"],
            stack_size=config["data"]["stack_size"],
            sequence_length=config["training"]["sequence_length"],
        )

        # Create dynamics model
        dynamics = DynamicsModel(
            latent_dim=config["vae"]["latent_dim"],
            num_actions=config["dynamics"]["num_actions"],
            hidden_size=config["dynamics"]["hidden_size"],
            num_layers=config["dynamics"]["num_layers"],
        )

        print(f"Model parameters: {sum(p.numel() for p in dynamics.parameters()):,}")

        # Create trainer
        trainer = DynamicsTrainer(
            dynamics_model=dynamics,
            vae_encoder=vae.encoder,
            train_dataset=dataset,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            batch_size=config["training"]["batch_size"],
            num_epochs=config["training"]["dynamics_epochs"],
            num_workers=config["training"]["num_workers"],
            scheduled_sampling_prob=config["training"]["scheduled_sampling_prob"],
            checkpoint_dir="./checkpoints",
            checkpoint_every=config["training"]["checkpoint_every"],
            device=device,
        )

        # Train
        trainer.train()

    else:
        raise ValueError(f"Unknown training stage: {args.stage}")


def cmd_dream(args):
    """Generate dream sequence."""
    from arcadedreamer.models.vae import VAE
    from arcadedreamer.models.dynamics import DynamicsModel
    from arcadedreamer.models.world_model import WorldModel
    from arcadedreamer.viz.dream import DreamGenerator
    from arcadedreamer.data.collector import AtariDataCollector

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on device: {device}")

    # Load VAE
    vae = VAE(
        in_channels=config["data"]["stack_size"],
        latent_dim=config["vae"]["latent_dim"],
        beta=config["vae"]["beta"],
    )

    vae_checkpoint = args.vae_checkpoint
    if vae_checkpoint is None:
        checkpoint_dir = Path("./checkpoints")
        vae_checkpoints = sorted(checkpoint_dir.glob("vae_epoch_*.pt"))
        if not vae_checkpoints:
            raise FileNotFoundError("No VAE checkpoint found.")
        vae_checkpoint = str(vae_checkpoints[-1])

    print(f"Loading VAE from {vae_checkpoint}")
    vae_state = torch.load(vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_state["model_state_dict"])

    # Load dynamics
    dynamics = DynamicsModel(
        latent_dim=config["vae"]["latent_dim"],
        num_actions=config["dynamics"]["num_actions"],
        hidden_size=config["dynamics"]["hidden_size"],
        num_layers=config["dynamics"]["num_layers"],
    )

    dyn_checkpoint = args.dynamics_checkpoint
    if dyn_checkpoint is None:
        checkpoint_dir = Path("./checkpoints")
        dyn_checkpoints = sorted(checkpoint_dir.glob("dynamics_epoch_*.pt"))
        if not dyn_checkpoints:
            raise FileNotFoundError("No dynamics checkpoint found.")
        dyn_checkpoint = str(dyn_checkpoints[-1])

    print(f"Loading dynamics from {dyn_checkpoint}")
    dyn_state = torch.load(dyn_checkpoint, map_location=device)
    dynamics.load_state_dict(dyn_state["model_state_dict"])

    # Create world model
    world_model = WorldModel(vae, dynamics)

    # Get seed frames from game
    print(f"Getting seed frames from {args.game}")
    collector = AtariDataCollector(
        game_name=args.game,
        save_dir="./temp_seed",
        frame_size=config["data"]["frame_size"],
        stack_size=config["data"]["stack_size"],
    )
    seed_frames = collector.get_seed_frames(num_frames=4, seed=args.seed)
    seed_tensor = torch.from_numpy(seed_frames)

    # Parse action sequence
    if args.actions:
        actions = [int(a.strip()) for a in args.actions.split(",")]
    else:
        # Default: repeat action 1 (usually fire)
        actions = [1]

    # Output path
    output_path = args.output
    if output_path is None:
        os.makedirs("./outputs", exist_ok=True)
        game_short = args.game.replace("NoFrameskip-v4", "")
        output_path = f"./outputs/dream_{game_short}.gif"

    # Generate dream
    print(f"Generating {config['viz']['dream_frames']} dream frames...")
    generator = DreamGenerator(world_model, device=device)
    generator.dream_and_save(
        seed_tensor,
        actions,
        output_path=output_path,
        num_frames=config["viz"]["dream_frames"],
        fps=config["viz"]["fps"],
    )

    print(f"\nDream saved to: {output_path}")


def cmd_play(args):
    """Interactive play mode."""
    from arcadedreamer.models.vae import VAE
    from arcadedreamer.models.dynamics import DynamicsModel
    from arcadedreamer.models.world_model import WorldModel
    from arcadedreamer.viz.interactive import InteractiveVisualizer

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on device: {device}")

    # Load VAE
    vae = VAE(
        in_channels=config["data"]["stack_size"],
        latent_dim=config["vae"]["latent_dim"],
        beta=config["vae"]["beta"],
    )

    vae_checkpoint = args.vae_checkpoint
    if vae_checkpoint is None:
        checkpoint_dir = Path("./checkpoints")
        vae_checkpoints = sorted(checkpoint_dir.glob("vae_epoch_*.pt"))
        if not vae_checkpoints:
            raise FileNotFoundError("No VAE checkpoint found.")
        vae_checkpoint = str(vae_checkpoints[-1])

    print(f"Loading VAE from {vae_checkpoint}")
    vae_state = torch.load(vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_state["model_state_dict"])

    # Load dynamics
    dynamics = DynamicsModel(
        latent_dim=config["vae"]["latent_dim"],
        num_actions=config["dynamics"]["num_actions"],
        hidden_size=config["dynamics"]["hidden_size"],
        num_layers=config["dynamics"]["num_layers"],
    )

    dyn_checkpoint = args.dynamics_checkpoint
    if dyn_checkpoint is None:
        checkpoint_dir = Path("./checkpoints")
        dyn_checkpoints = sorted(checkpoint_dir.glob("dynamics_epoch_*.pt"))
        if not dyn_checkpoints:
            raise FileNotFoundError("No dynamics checkpoint found.")
        dyn_checkpoint = str(dyn_checkpoints[-1])

    print(f"Loading dynamics from {dyn_checkpoint}")
    dyn_state = torch.load(dyn_checkpoint, map_location=device)
    dynamics.load_state_dict(dyn_state["model_state_dict"])

    # Create world model
    world_model = WorldModel(vae, dynamics)

    # Create visualizer
    visualizer = InteractiveVisualizer(
        world_model=world_model,
        game_name=args.game,
        device=device,
        frame_size=config["data"]["frame_size"],
        display_scale=config["viz"]["display_scale"],
        fps=30,
    )

    # Run interactive mode
    visualizer.run()


def main():
    parser = argparse.ArgumentParser(
        description="ArcadeDreamer - Atari World Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect data from all configured games
  python main.py collect --frames 50000

  # Collect from a specific game
  python main.py collect --game BreakoutNoFrameskip-v4 --frames 10000

  # Train VAE
  python main.py train --stage vae

  # Train dynamics model
  python main.py train --stage dynamics

  # Generate dream sequence
  python main.py dream --game BreakoutNoFrameskip-v4 --actions "1,1,2,2,3,3"

  # Interactive play mode
  python main.py play --game BreakoutNoFrameskip-v4
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="arcadedreamer/configs/default.yaml",
        help="Path to config file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Collect command
    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect training data from Atari games",
    )
    collect_parser.add_argument(
        "--game",
        type=str,
        help="Specific game to collect from (default: all in config)",
    )
    collect_parser.add_argument(
        "--frames",
        type=int,
        help="Number of frames to collect per game",
    )
    collect_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train VAE or dynamics model",
    )
    train_parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["vae", "dynamics"],
        help="Training stage: 'vae' or 'dynamics'",
    )
    train_parser.add_argument(
        "--vae-checkpoint",
        type=str,
        help="VAE checkpoint path (for dynamics training)",
    )

    # Dream command
    dream_parser = subparsers.add_parser(
        "dream",
        help="Generate dream sequence as GIF",
    )
    dream_parser.add_argument(
        "--game",
        type=str,
        required=True,
        help="Game to dream about",
    )
    dream_parser.add_argument(
        "--actions",
        type=str,
        help='Comma-separated action sequence (e.g., "1,1,2,3")',
    )
    dream_parser.add_argument(
        "--output",
        type=str,
        help="Output GIF path",
    )
    dream_parser.add_argument(
        "--vae-checkpoint",
        type=str,
        help="VAE checkpoint path",
    )
    dream_parser.add_argument(
        "--dynamics-checkpoint",
        type=str,
        help="Dynamics checkpoint path",
    )
    dream_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for seed frame selection",
    )

    # Play command
    play_parser = subparsers.add_parser(
        "play",
        help="Interactive play mode with side-by-side visualization",
    )
    play_parser.add_argument(
        "--game",
        type=str,
        required=True,
        help="Game to play",
    )
    play_parser.add_argument(
        "--vae-checkpoint",
        type=str,
        help="VAE checkpoint path",
    )
    play_parser.add_argument(
        "--dynamics-checkpoint",
        type=str,
        help="Dynamics checkpoint path",
    )

    args = parser.parse_args()

    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "dream":
        cmd_dream(args)
    elif args.command == "play":
        cmd_play(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
