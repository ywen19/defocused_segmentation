#!/bin/bash
# run_extraction.sh - Simple script to run the frame extraction

# Exit on error
set -e

# Configuration
INPUT_DIR="../data/video_defocused"
OUTPUT_DIR="../data/video_defocused_processed"
PYTHON_SCRIPT="frame_extraction.py"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}====================================${NC}"
echo -e "${BLUE}    Defocused Video Processing      ${NC}"
echo -e "${BLUE}====================================${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 is not installed. Please install Python 3.7 or higher.${NC}"
    exit 1
fi

# Install required packages
echo -e "${YELLOW}Installing required packages...${NC}"
pip install numpy opencv-python pillow torch tqdm

# Check if CUDA is available
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${GREEN}CUDA is available! GPU acceleration will be used.${NC}"
    
    # Get GPU information
    python3 -c "import torch; print(f'Found {torch.cuda.device_count()} GPUs.')"
    python3 -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
    
    # Use GPU for processing
    GPU_FLAG="--gpu"
else
    echo -e "${YELLOW}CUDA is not available. CPU processing will be used.${NC}"
    GPU_FLAG=""
fi

# Verify input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}Input directory does not exist: $INPUT_DIR${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start time
START_TIME=$(date +%s)

# Run the script for both train and test
echo -e "${GREEN}Starting frame extraction...${NC}"
python3 "$PYTHON_SCRIPT" --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --dataset both $GPU_FLAG

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Format duration
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

echo -e "${GREEN}Processing completed in ${HOURS}h ${MINUTES}m ${SECONDS}s!${NC}"
echo -e "${GREEN}Output saved to: $OUTPUT_DIR${NC}"

# Show resource usage
echo -e "${YELLOW}Resource usage summary:${NC}"
echo -e "${YELLOW}CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')%${NC}"
echo -e "${YELLOW}Memory Usage: $(free -h | awk '/^Mem/ {print $3 " / " $2}')${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}GPU Memory Usage: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{print $1"MB / "$2"MB"}')${NC}"
fi

echo -e "${GREEN}Done!${NC}"