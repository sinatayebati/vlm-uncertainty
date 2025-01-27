#!/bin/bash

# Exit on error
set -e

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

# Parse command line arguments
MODE=${1:-vlm}  # Default to 'vlm' if no argument provided

# Set paths based on mode
if [ "$MODE" = "llm" ]; then
    RESULT_PATH="output_llm"
    SAVE_PATH="trained_policies_llm"
else
    RESULT_PATH="output"
    SAVE_PATH="trained_policies"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p "$SAVE_PATH"
mkdir -p logs

# Run training
echo "Starting training in $MODE mode..."
python -m src.training.train_policy \
    --result_data_path "$RESULT_PATH" \
    --save_model_path "$SAVE_PATH" \
    --config_file "configs/training_config.yaml" \
    --mode "$MODE" 2>&1

echo "Training completed!"