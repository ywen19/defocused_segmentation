#!/bin/bash
# run_frame_extraction.sh - Frame extraction runner using multiprocessing (CPU only)

# Exit on error
set -e

# Configuration
INPUT_DIR="../data/video_defocused"
OUTPUT_DIR="../data/video_defocused_processed"
PYTHON_SCRIPT="frame_extraction.py"
RATIO="${RATIO:-0.2}"  # Default ratio

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}====================================${NC}"
echo -e "${BLUE}    Defocused Video Processing      ${NC}"
echo -e "${BLUE}====================================${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 is not installed.${NC}"
    exit 1
fi

echo -e "${YELLOW}Installing required packages...${NC}"
pip install numpy opencv-python pillow torch tqdm

mkdir -p "$OUTPUT_DIR"

START_TIME=$(date +%s)

echo -e "${GREEN}Starting frame extraction at $RATIO sampling ratio...${NC}"
python3 "$PYTHON_SCRIPT" --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --dataset both --ratio "$RATIO"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo -e "${GREEN}Completed in ${HOURS}h ${MINUTES}m ${SECONDS}s!${NC}"
echo -e "${GREEN}Output saved to: $OUTPUT_DIR${NC}"

echo -e "${YELLOW}Resource usage summary:${NC}"
echo -e "${YELLOW}CPU Usage: $(top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}')%${NC}"
MEMORY_USAGE=$(free -h | awk '/^Mem/ {print $3 " / " $2}')
echo -e "${YELLOW}Memory Usage: $MEMORY_USAGE${NC}"

echo -e "${GREEN}Done!${NC}"
