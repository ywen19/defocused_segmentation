#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Parse command line arguments
MODE="first_frame"  # Default mode
if [ $# -ge 1 ]; then
    if [ "$1" = "first_frame" ] || [ "$1" = "all_frames" ]; then
        MODE="$1"
    else
        echo "Error: Invalid mode. Use 'first_frame' or 'all_frames'"
        exit 1
    fi
fi

# Activate environment
source ~/.bashrc
conda activate sam2_matanyone

echo "[INFO] Starting mask extraction in ${MODE} mode at $(date)"

# Optional memory fragmentation handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Script and data paths
SCRIPT_PATH="yolo_sam2_mask_extract.py"
DATA_DIR="../data/video_defocused_processed"

# Create timestamped log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Run the script with the specified mode
LOG_FILE="${LOG_DIR}/mask_extraction_${MODE}_${TIMESTAMP}.log"
echo "[INFO] Running ${MODE} mask extraction..."
python "$SCRIPT_PATH" --data_dir "$DATA_DIR" --mode "$MODE" 2>&1 | tee "$LOG_FILE"
echo "[INFO] Completed ${MODE} mask extraction at $(date)"

echo "[INFO] Finished mask extraction task at $(date)"

# Print summary of results
echo "=== Extraction Summary ==="
echo "Mode: $MODE"
echo "Log file: $LOG_FILE"
echo "=========================="