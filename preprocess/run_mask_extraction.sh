#!/bin/bash
set -e  # 遇到错误立即退出

# Activate environment
source ~/.bashrc
conda activate sam2_matanyone

echo "[INFO] Starting mask extraction at $(date)"

# Optional memory fragmentation handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_PATH="yolo_sam2_mask_extract.py"
DATA_DIR="../data/video_defocused_processed"
LOG_FILE="run_mask_extraction_$(date +%Y%m%d_%H%M%S).log"

# Execute with logging
python "$SCRIPT_PATH" --data_dir "$DATA_DIR" | tee "$LOG_FILE"

echo "[INFO] Finished mask extraction at $(date)"
