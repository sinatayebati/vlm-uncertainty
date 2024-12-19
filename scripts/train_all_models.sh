#!/bin/bash

# Exit on error
set -e

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

# Create necessary directories
echo "Creating directories..."
mkdir -p trained_policies
mkdir -p logs

# Run training
echo "Starting training..."
python -m src.training.train_policy \
    --result_data_path "output" \
    --save_model_path "trained_policies" \
    --config_file "configs/training_config.yaml" 2>&1

echo "Training completed!"