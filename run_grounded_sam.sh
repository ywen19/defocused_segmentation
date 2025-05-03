#!/bin/bash
# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate grounding_matting

# Set PYTHONPATH for GroundingDINO
CURRENT_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR/Grounded-Segment-Anything/GroundingDINO

# Run the demo
cd Grounded-Segment-Anything
python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo1.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "bear" \
  --device "cuda"
