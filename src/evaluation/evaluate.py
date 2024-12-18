import os
import json
import pickle
from typing import Dict, Any, List
import argparse
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.models.conformal import Abstention_CP
from src.utils.metrics import (
    compute_base_metrics,
    compute_calibration_error,
    compute_set_metrics
)
from data_utils import DATASETS, SEEDBENCH_CATS, OODCV_CATS

def load_policy(policy_path: str) -> Dict[str, Any]:
    """Load trained policy and its hyperparameters."""
    checkpoint = torch.load(policy_path)
    return checkpoint['hyperparameters']

def evaluate_model(
    result_data: List[Dict],
    policy_params: Dict[str, float],
    cal_ratio: float = 0.5
) -> Dict[str, float]:
    """Evaluate a model using the trained policy parameters."""
    # Split data into calibration and test sets
    cal_result_data, test_result_data = train_test_split(
        result_data, train_size=cal_ratio, random_state=42
    )

    # Create test_id_to_answer mapping
    test_id_to_answer = {str(row["id"]): row["answer"] for row in test_result_data}

    # Compute base metrics
    acc, E_ratio, F_ratio = compute_base_metrics(test_result_data)

    # Compute calibration errors
    ece = compute_calibration_error(result_data, norm='l1')
    mce = compute_calibration_error(result_data, norm='max')

    # Get prediction sets using Abstention_CP
    pred_outputs = Abstention_CP(cal_result_data, test_result_data, policy_params)

    # Compute set-based metrics
    set_metrics = compute_set_metrics(pred_outputs, test_id_to_answer)

    # Combine all metrics
    metrics = {
        'acc': acc,
        'E_ratio': E_ratio,
        'F_ratio': F_ratio,
        'ece': ece,
        'mce': mce,
        **set_metrics
    }

    return metrics

def main(args):
    results = {}
    model_names = os.listdir(args.result_data_path)

    for model_name in tqdm(model_names, desc="Processing models"):
        model_results = {}
        policy_base_path = Path(args.policy_path) / model_name
        
        if args.mode == "all":
            # Evaluate on all datasets
            for dataset_name in DATASETS:
                # Load corresponding policy for this model and dataset
                policy_file = f"{model_name}_{dataset_name}_policy.pth"
                policy_path = policy_base_path / policy_file
                
                if not policy_path.exists():
                    print(f"Warning: Policy not found for {model_name} on {dataset_name}")
                    continue
                
                file_path = Path(args.result_data_path) / model_name / f"{dataset_name}.pkl"
                if not file_path.exists():
                    continue
                    
                try:
                    policy_params = load_policy(str(policy_path))
                    with open(file_path, 'rb') as f:
                        result_data = pickle.load(f)
                    
                    metrics = evaluate_model(result_data, policy_params, args.cal_ratio)
                    model_results[dataset_name] = metrics
                except Exception as e:
                    print(f"Error processing {model_name} - {dataset_name}: {str(e)}")
                
        elif args.mode == "seedbench":
            # Evaluate on SeedBench categories
            policy_file = f"{model_name}_seedbench_policy.pth"
            policy_path = policy_base_path / policy_file

            if not policy_path.exists():
                    print(f"Warning: Policy not found for {model_name} on seedbench")
                    continue

            file_path = Path(args.result_data_path) / model_name / "seedbench.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    result_data = pickle.load(f)
                    
                for cat_id in range(1, 10):
                    cat_data = [row for row in result_data if row['question_type_id'] == cat_id]
                    if cat_data:
                        metrics = evaluate_model(cat_data, policy_params, args.cal_ratio)
                        model_results[f"category_{cat_id}"] = metrics
                        
        elif args.mode == "oodcv":
            policy_file = f"{model_name}_oodcv_policy.pth"
            policy_path = policy_base_path / policy_file

            if not policy_path.exists():
                    print(f"Warning: Policy not found for {model_name} on oodcv")
                    continue

            # Evaluate on OODCV categories
            file_path = Path(args.result_data_path) / model_name / "oodcv.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    result_data = pickle.load(f)
                    
                for situation in OODCV_CATS:
                    cat_data = [row for row in result_data if row['situation'] == situation]
                    if cat_data:
                        metrics = evaluate_model(cat_data, policy_params, args.cal_ratio)
                        model_results[situation] = metrics

        results[model_name] = model_results


    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_data_path", type=str, required=True,
                        help="Path to result data")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="Base path to trained policies directory")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save evaluation results")
    parser.add_argument("--mode", choices=["all", "seedbench", "oodcv"],
                        default="all", help="Evaluation mode")
    parser.add_argument("--cal_ratio", type=float, default=0.5,
                        help="Ratio of calibration data")
    
    args = parser.parse_args()
    main(args)