#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Setting up environment for SAM2 and MatAnyone with CUDA 12.4 support via PyTorch wheels..."

# Create conda environment from YAML file
conda env create -f environment.yml

# Activate environment
eval "$(conda shell.bash hook)"
conda activate sam2_matanyone

# Clone repositories
echo "Cloning repositories..."
git clone https://github.com/facebookresearch/sam2.git
git clone https://github.com/pq-yang/MatAnyone.git

# Install SAM2
echo "Installing SAM2..."
cd sam2
# Install with development dependencies
pip install -e ".[dev]"
cd ..

# Install MatAnyone
echo "Installing MatAnyone..."
cd MatAnyone
pip install -e .
cd ..

# Create directories for model weights
mkdir -p sam2/weights
mkdir -p MatAnyone/pretrained_models

# Set CUDA_HOME environment variable
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo "Installation complete! Now you need to download the models:"
echo "- For SAM2: You can download models from Hugging Face directly in code or manually from the GitHub repository."
echo "- For MatAnyone: Models will be automatically downloaded during first inference or can be manually downloaded from their releases page."
echo ""
echo "Environment is configured for CUDA 12.4"
echo "To activate the environment, run: conda activate sam2_matanyone"