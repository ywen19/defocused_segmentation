#!/bin/bash
# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate grounding_matting

# Run MatAnyone inference
cd MatAnyone
python inference_hf.py \
  --input_path "inputs/video/test-sample1.mp4" \
  --mask_path "inputs/mask/test-sample1.png" \
  --output_path "outputs"
