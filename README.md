# ArcadeDreamer

A 2D world model that learns to simulate Atari games from scratch. ArcadeDreamer uses a Variational Autoencoder (VAE) to compress game frames into a compact latent space, and a GRU-based dynamics model to predict future states given actions.

<p align="center">
  <img src="https://stable-baselines.readthedocs.io/en/master/_images/breakout.gif" alt="Breakout Gameplay">
</p>

## Features

- **Data Collection**: Collect training data from Atari games using Gymnasium and ale-py
- **VAE Training**: Learn compressed latent representations of game frames
- **Dynamics Model**: Predict future game states in latent space
- **Dream Mode**: Generate imagined game sequences as animated GIFs
- **Interactive Mode**: Play games with real-time model predictions displayed side-by-side

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ArcadeDreamer                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Encoder    │    │   Dynamics   │    │   Decoder    │      │
│  │  (4×64×64)   │───▶│    (GRU)     │───▶│  (64→4×64×64)│      │
│  │    → 64      │    │   z + a → z' │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         └───────────────────┴───────────────────┘               │
│                    World Model                                  │
└─────────────────────────────────────────────────────────────────┘
```

### VAE Encoder
- Input: 4 stacked grayscale frames (4×64×64)
- 4 convolutional layers: 32→64→128→256 channels
- Fully connected: 4096→512→64 (mu, logvar)
- Output: 64-dimensional latent vector

### VAE Decoder
- Input: 64-dimensional latent vector
- Fully connected: 64→512→4096
- 4 transposed convolutions: 256→128→64→32→4
- Output: Reconstructed frames (4×64×64)

### Dynamics Model
- Input: Latent (64) + one-hot action (18) = 82
- GRU with 256 hidden units
- Output: Predicted next latent (64)

## Installation

```bash
# Clone the repository
cd arcadedreamer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Atari ROMs (required for ale-py)
# Option 1: Use AutoROM (easiest)
pip install autorom
autorom --accept-license

# Option 2: Manual ROM installation
# Download ROMs and place in ~/.ale/roms/
```

## Usage

### 1. Collect Training Data

Collect 50,000 frames from each configured game (Breakout, Pong, SpaceInvaders):

```bash
python main.py collect --frames 50000
```

Collect from a specific game:

```bash
python main.py collect --game BreakoutNoFrameskip-v4 --frames 10000
```

### 2. Train the VAE

Train the VAE for 50 epochs to learn latent representations:

```bash
python main.py train --stage vae
```

Checkpoints are saved to `./checkpoints/` every 10 epochs.

### 3. Train the Dynamics Model

Train the dynamics model for 100 epochs (requires trained VAE):

```bash
python main.py train --stage dynamics
```

Or specify a specific VAE checkpoint:

```bash
python main.py train --stage dynamics --vae-checkpoint ./checkpoints/vae_epoch_50.pt
```

### 4. Generate Dreams

Generate a dream sequence as an animated GIF:

```bash
python main.py dream --game BreakoutNoFrameskip-v4 --actions "1,1,1,2,2,3,3"
```

The `--actions` parameter specifies the action sequence:
- 0: NOOP
- 1: FIRE
- 2: RIGHT
- 3: LEFT
- 4: UP
- 5: DOWN

Output is saved to `./outputs/dream_<game>.gif`.

### 5. Interactive Play Mode

Play the game with real-time model predictions:

```bash
python main.py play --game BreakoutNoFrameskip-v4
```

Controls:
- Arrow keys: Move
- Space: Fire
- R: Reset
- ESC: Quit

The display shows three panels:
1. **Real**: Actual game frame
2. **Predicted**: Model's prediction
3. **Overlay**: Comparison (Green=real, Red=predicted, Yellow=overlap)

## Configuration

Edit `arcadedreamer/configs/default.yaml` to customize:

```yaml
# Data collection
data:
  games:
    - BreakoutNoFrameskip-v4
    - PongNoFrameskip-v4
    - SpaceInvadersNoFrameskip-v4
  frames_per_game: 50000
  frame_size: 64
  stack_size: 4

# VAE architecture
vae:
  latent_dim: 64
  beta: 0.0001

# Dynamics model
dynamics:
  hidden_size: 256
  num_actions: 18

# Training
training:
  vae_epochs: 50
  dynamics_epochs: 100
  batch_size: 128
  lr: 0.0003
  scheduled_sampling_prob: 0.5
```

## Project Structure

```
arcadedreamer/
├── main.py                    # CLI entry point
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── arcadedreamer/
│   ├── __init__.py
│   ├── configs/
│   │   └── default.yaml       # Default configuration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py       # Data collection
│   │   ├── dataset.py         # PyTorch datasets
│   │   └── preprocessing.py   # Frame preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vae.py             # VAE model
│   │   ├── dynamics.py        # GRU dynamics model
│   │   └── world_model.py     # Combined world model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── vae_trainer.py     # VAE training loop
│   │   ├── dynamics_trainer.py # Dynamics training
│   │   └── utils.py           # Training utilities
│   └── viz/
│       ├── __init__.py
│       ├── dream.py           # Dream GIF generation
│       └── interactive.py     # Pygame visualization
├── checkpoints/               # Model checkpoints
├── data/collected/            # Collected training data
└── outputs/                   # Generated GIFs
```

## Training Pipeline

```bash
# Complete training pipeline
python main.py collect --frames 50000    # ~30 min per game
python main.py train --stage vae         # ~2-4 hours
python main.py train --stage dynamics    # ~4-8 hours

# Generate results
python main.py dream --game BreakoutNoFrameskip-v4
python main.py play --game BreakoutNoFrameskip-v4
```

## Technical Details

### Training Parameters
- **VAE**: 50 epochs, MSE reconstruction + KL divergence (beta=0.0001)
- **Dynamics**: 100 epochs, MSE on latent predictions, scheduled sampling (50%)
- **Optimizer**: Adam (lr=3e-4, weight_decay=1e-5)
- **Scheduler**: Cosine annealing
- **Batch size**: 128
- **Gradient clipping**: max_norm=1.0

### Frame Preprocessing
1. Convert RGB to grayscale
2. Resize to 64x64 using INTER_AREA interpolation
3. Normalize to [0, 1]
4. Stack 4 consecutive frames

### Action Space
Uses 18 actions (ALE maximum) for consistency across games:
- 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
- 6-17: Combinations and diagonals

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended for training)
- ~10GB disk space for data
- ~4GB GPU memory for training

## License

MIT License

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/) for the Atari environments
- [ALE (Arcade Learning Environment)](https://github.com/mgbellemare/Arcade-Learning-Environment) for Atari emulation
- Inspired by [World Models](https://worldmodels.github.io/) by Ha & Schmidhuber
