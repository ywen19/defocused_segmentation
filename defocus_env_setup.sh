#!/bin/bash

set -e  # Stop on error

echo "Creating Conda environment with Python 3.12 + CUDA 12.1"
conda env create -f defocus_environment.yml

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate defocused-env

echo "Cloning MiDaS repository..."
if [ ! -d "MiDaS" ]; then
  git clone https://github.com/isl-org/MiDaS.git
else
  echo "MiDaS already cloned."
fi

echo "Verifying CUDA and PyTorch:"
python -c "import torch; print('Torch CUDA Available:', torch.cuda.is_available()); print('Torch CUDA Version:', torch.version.cuda)"

echo "Setup complete!"
echo "To activate your environment later, run:"
echo "conda activate defocused-env"
