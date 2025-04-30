#!/bin/bash
set -e

# Configuration
MAX_RETRIES=50
LOG_FILE="./processing_log.txt"
SCRIPT_PATH="./preprocess/dof_blur.py"

echo "▶ Starting video processing with improved error handling..."
echo "$(date): Processing started" >> $LOG_FILE

for i in $(seq 1 $MAX_RETRIES); do
    echo "🔁 Pass $i of $MAX_RETRIES..."
    
    # Set +e to prevent the script from exiting when the Python script returns non-zero
    set +e
    
    # Run the actual script
    python $SCRIPT_PATH
    EXIT_CODE=$?
    
    # Set -e again to make the script exit on error
    set -e
    
    # Check if script exited with return code 0 (success)
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Processing completed successfully!"
        echo "$(date): Processing completed successfully" >> $LOG_FILE
        exit 0
    fi
    
    # Script exited with error or needs restart
    echo "$(date): Pass $i finished with exit code $EXIT_CODE" >> $LOG_FILE
    
    if [ $EXIT_CODE -eq 1 ]; then
        echo "🔁 Restarting after 10s..."
        sleep 10
    else
        echo "❌ Script failed with unexpected exit code $EXIT_CODE"
        echo "$(date): Script failed with exit code $EXIT_CODE" >> $LOG_FILE
        exit $EXIT_CODE
    fi
done

echo "❌ Maximum retries reached. Please check logs."
echo "$(date): Maximum retries reached" >> $LOG_FILE
exit 1