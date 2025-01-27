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
    POLICY_PATH="trained_policies_llm"
else
    RESULT_PATH="output"
    POLICY_PATH="trained_policies"
fi

# Create necessary directories
mkdir -p evaluation_results
mkdir -p logs

# Run evaluation
echo "Starting evaluation in $MODE mode..."

# Evaluate on all datasets
python -m src.evaluation.evaluate \
    --result_data_path "$RESULT_PATH" \
    --policy_path "$POLICY_PATH" \
    --output_file "evaluation_results/all_results.json" \
    --mode "$MODE" 2>&1 | tee logs/eval_all_$(date +%Y%m%d_%H%M%S).log

# Evaluate on all datasets using literature baseline method
python -m src.evaluation.baseline_evaluate \
    --result_data_path "$RESULT_PATH" \
    --file_to_write "evaluation_results/all_results_literature_baseline.json" \
    --mode "$MODE"

# # Evaluate on SeedBench
# python -m src.evaluation.evaluate \
#     --result_data_path "output" \
#     --policy_path "trained_policies" \
#     --output_file "evaluation_results/seedbench_results.json" \
#     --mode "seedbench" 2>&1 | tee logs/eval_seedbench_$(date +%Y%m%d_%H%M%S).log

# # Evaluate on OODCV
# python -m src.evaluation.evaluate \
#     --result_data_path "output" \
#     --policy_path "trained_policies" \
#     --output_file "evaluation_results/oodcv_results.json" \
#     --mode "oodcv" 2>&1 | tee logs/eval_oodcv_$(date +%Y%m%d_%H%M%S).log

echo "Evaluation completed!"