# Running ArcadeDreamer on RunPod with H100

## Quick Start

### 1. Create a RunPod Instance

1. Go to [RunPod.io](https://runpod.io)
2. Select **GPU Cloud** → **Secure Cloud**
3. Choose **NVIDIA H100 80GB** (or H100 SXM for fastest)
4. Select a PyTorch template (e.g., `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`)
5. Add a **Network Volume** (50GB+) to persist data between sessions
6. Launch the pod

### 2. Setup on RunPod

SSH into your pod or use the web terminal:

```bash
# Clone repo
git clone https://github.com/VishnuMano/arcadedreamer.git
cd arcadedreamer

# Install dependencies
pip install -r requirements.txt
pip install autorom && autorom --accept-license

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 3. Collect Data (on GPU - faster)

```bash
# Collect from all 3 games (takes ~10 min on H100)
python main.py collect --frames 50000

# Or single game for testing
python main.py collect --game BreakoutNoFrameskip-v4 --frames 50000
```

### 4. Train Models

```bash
# Train VAE (~10-15 min on H100 vs hours on CPU)
python runpod_train.py --stage vae --epochs 50

# Train dynamics (~20-30 min on H100)
python runpod_train.py --stage dynamics --epochs 100
```

### 5. Download Results

```bash
# From your local machine, use scp or runpodctl
runpodctl receive checkpoints/
runpodctl receive outputs/

# Or zip and download via web UI
zip -r results.zip checkpoints/ outputs/
```

## Performance Comparison

| Hardware | VAE (50 epochs) | Dynamics (100 epochs) |
|----------|-----------------|----------------------|
| CPU (M1) | ~4-6 hours | ~8-12 hours |
| RTX 3090 | ~30-45 min | ~60-90 min |
| A100 40GB | ~15-20 min | ~30-40 min |
| **H100 80GB** | **~8-12 min** | **~15-25 min** |

## Optimized Settings for H100

The `runpod_train.py` script automatically:
- Enables TF32 for faster matrix operations
- Uses larger batch sizes (512 vs 128)
- Uses `torch.compile()` for kernel optimization
- Uses 8 data loader workers

### Manual Tuning

```bash
# Maximum performance (if memory allows)
python runpod_train.py --stage vae --batch-size 1024 --epochs 50

# More frequent checkpoints
python runpod_train.py --stage vae --save-every 5
```

## Using Network Volumes

To persist data between pod restarts:

1. Create a Network Volume in RunPod dashboard
2. Mount it at `/workspace/data` when creating the pod
3. Use `--data-dir /workspace/data/collected` and `--checkpoint-dir /workspace/data/checkpoints`

```bash
# Collect to persistent volume
python main.py collect --frames 50000
mv data/collected /workspace/data/

# Train from persistent volume
python runpod_train.py --stage vae \
    --data-dir /workspace/data/collected \
    --checkpoint-dir /workspace/data/checkpoints
```

## Cost Estimate

| GPU | Price/hr | Full Training | Cost |
|-----|----------|---------------|------|
| H100 SXM | ~$4.00 | ~30 min | ~$2.00 |
| H100 PCIe | ~$3.00 | ~40 min | ~$2.00 |
| A100 80GB | ~$2.00 | ~60 min | ~$2.00 |

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python runpod_train.py --stage vae --batch-size 256
```

### Slow Data Loading
```bash
# Ensure data is on SSD, not network volume for training
cp -r /workspace/data/collected ./data/
```

### CUDA Errors
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```
