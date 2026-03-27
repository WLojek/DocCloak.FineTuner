#!/bin/bash
# Setup script for GPU cloud environments (RunPod, Vast.ai, etc.)
# Creates a venv, installs dependencies, then reinstalls PyTorch with the correct CUDA version.
#
# Usage: bash setup_gpu.sh

set -e

# Step 1: Create and activate venv
python -m venv .venv
source .venv/bin/activate

# Step 2: Install all dependencies
pip install -e .

# Step 3: Detect CUDA driver and reinstall torch with correct version
if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
    MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    echo ""
    echo "Detected CUDA driver: $CUDA_VERSION"

    # Pick the right PyTorch CUDA index
    if [ "$MAJOR" -ge 13 ]; then
        INDEX="https://download.pytorch.org/whl/cu126"
    elif [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 4 ]; then
        INDEX="https://download.pytorch.org/whl/cu124"
    elif [ "$MAJOR" -eq 12 ]; then
        INDEX="https://download.pytorch.org/whl/cu121"
    elif [ "$MAJOR" -eq 11 ]; then
        INDEX="https://download.pytorch.org/whl/cu118"
    else
        echo "Unsupported CUDA version: $CUDA_VERSION"
        exit 1
    fi

    echo "Reinstalling PyTorch from: $INDEX"
    pip install torch torchvision torchaudio --index-url "$INDEX" --force-reinstall
else
    echo "No NVIDIA GPU detected. Using default (CPU/MPS) torch."
fi

echo ""
echo "Setup complete. Verifying..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To initialize training workspace:"
echo "  doccloak-finetune init -c config.herbert.yaml -o workspace_herbert"
echo ""
echo "To start training:"
echo "  doccloak-finetune run -c config.herbert.yaml -o workspace_herbert"
