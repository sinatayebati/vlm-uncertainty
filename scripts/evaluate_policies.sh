#!/bin/bash

# Exit on error
set -e

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

# Create necessary directories
mkdir -p evaluation_results
mkdir -p logs

# Run evaluation
echo "Starting evaluation..."

# Evaluate on all datasets
python -m src.evaluation.evaluate \
    --result_data_path "output" \
    --policy_path "trained_policies" \
    --output_file "evaluation_results/all_results.json" \
    --mode "all" 2>&1 | tee logs/eval_all_$(date +%Y%m%d_%H%M%S).log

# Evaluate on SeedBench
python -m src.evaluation.evaluate \
    --result_data_path "output" \
    --policy_path "trained_policies" \
    --output_file "evaluation_results/seedbench_results.json" \
    --mode "seedbench" 2>&1 | tee logs/eval_seedbench_$(date +%Y%m%d_%H%M%S).log

# Evaluate on OODCV
python -m src.evaluation.evaluate \
    --result_data_path "output" \
    --policy_path "trained_policies" \
    --output_file "evaluation_results/oodcv_results.json" \
    --mode "oodcv" 2>&1 | tee logs/eval_oodcv_$(date +%Y%m%d_%H%M%S).log

echo "Evaluation completed!"