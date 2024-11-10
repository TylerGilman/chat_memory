#!/bin/bash
set -e

echo "=== System Information ==="
nvidia-smi

# Set up environment variables
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export HF_HOME=/workspace/cache
export TRANSFORMERS_CACHE=/workspace/cache
export TORCH_EXTENSIONS_DIR=/workspace/cache

# Create necessary directories
mkdir -p models cache

# Start training
echo "=== Starting Training ==="
python3 training/train.py
