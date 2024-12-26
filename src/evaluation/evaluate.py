import os
import json
import pickle
from typing import Dict, Any, List, Union, Tuple
import argparse
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.models.conformal import Abstention_CP
from src.utils.metrics import (
    compute_calibration_error,
    compute_set_metrics
)
from data_utils import DATASETS, SEEDBENCH_CATS, OODCV_CATS

BASELINE_PARAMS = {
    'alpha': 0.1,
    'beta': 0.2,
    'cum_prob_threshold': 0.9
}

def load_policy(policy_path: str) -> Dict[str, Any]:
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    checkpoint = torch.load(policy_path)
    return checkpoint['hyperparameters']

def evaluate_model(
    result_data: List[Dict],
    policy_params: Dict[str, float],
    cal_ratio: float = 0.5
) -> Dict[str, float]:
    """Evaluate a model using given parameters."""
    # Split data into calibration and test sets
    cal_result_data, test_result_data = train_test_split(
        result_data, train_size=cal_ratio, random_state=42
    )

    # Create test_id_to_answer mapping
    test_id_to_answer = {str(row["id"]): row["answer"] for row in test_result_data}

    # Compute calibration errors
    ece = compute_calibration_error(result_data, norm='l1')
    mce = compute_calibration_error(result_data, norm='max')

    # Get prediction sets using Abstention_CP
    pred_outputs, accuracy, abstention_rate, average_set_size = Abstention_CP(
        cal_result_data, test_result_data, policy_params
    )

    # Compute set-based metrics
    set_metrics = compute_set_metrics(pred_outputs, test_id_to_answer)

    # Combine all metrics
    metrics = {
        'ece': ece,
        'mce': mce,
        **set_metrics
    }

    return metrics


def evaluate_with_both_params(
    result_data: List[Dict],
    optimized_params: Dict[str, float],
    cal_ratio: float = 0.5
) -> Tuple[Dict[str, float], Dict[str, float]]:

    optimized_metrics = evaluate_model(result_data, optimized_params, cal_ratio)
    baseline_metrics = evaluate_model(result_data, BASELINE_PARAMS, cal_ratio)
    
    return optimized_metrics, baseline_metrics


def main(args):
    optimized_results = {}
    baseline_results = {}
    model_names = os.listdir(args.result_data_path)

    for model_name in tqdm(model_names, desc="Processing models"):
        optimized_model_results = {}
        baseline_model_results = {}
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
                    optimized_params = load_policy(str(policy_path))
                    with open(file_path, 'rb') as f:
                        result_data = pickle.load(f)
                    
                    opt_metrics, base_metrics = evaluate_with_both_params(
                        result_data, optimized_params, args.cal_ratio
                    )
                    
                    optimized_model_results[dataset_name] = opt_metrics
                    baseline_model_results[dataset_name] = base_metrics
                    
                except Exception as e:
                    print(f"Error processing {model_name} - {dataset_name}: {str(e)}")
                
        elif args.mode == "seedbench":
            policy_file = f"{model_name}_seedbench_policy.pth"
            policy_path = policy_base_path / policy_file

            if not policy_path.exists():
                    print(f"Warning: Policy not found for {model_name} on seedbench")
                    continue

            file_path = Path(args.result_data_path) / model_name / "seedbench.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    result_data = pickle.load(f)

            try:
                optimized_params = load_policy(str(policy_path))
                for cat_id in range(1, 10):
                    cat_data = [row for row in result_data if row['question_type_id'] == cat_id]
                    if cat_data:
                        opt_metrics, base_metrics = evaluate_with_both_params(
                            cat_data, optimized_params, args.cal_ratio
                        )
                        optimized_model_results[f"category_{cat_id}"] = opt_metrics
                        baseline_model_results[f"category_{cat_id}"] = base_metrics
            except Exception as e:
                    print(f"Error processing {model_name} - seedbench: {str(e)}")
                        
        elif args.mode == "oodcv":
            policy_file = f"{model_name}_oodcv_policy.pth"
            policy_path = policy_base_path / policy_file

            if not policy_path.exists():
                    print(f"Warning: Policy not found for {model_name} on oodcv")
                    continue
            
            file_path = Path(args.result_data_path) / model_name / "oodcv.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    result_data = pickle.load(f)
            try:
                optimized_params = load_policy(str(policy_path))
                for situation in OODCV_CATS:
                    cat_data = [row for row in result_data if row['situation'] == situation]
                    if cat_data:
                        opt_metrics, base_metrics = evaluate_with_both_params(
                            cat_data, optimized_params, args.cal_ratio
                        )
                        optimized_model_results[f"category_{cat_id}"] = opt_metrics
                        baseline_model_results[f"category_{cat_id}"] = base_metrics
            except Exception as e:
                    print(f"Error processing {model_name} - oodcv: {str(e)}")

        optimized_results[model_name] = optimized_model_results
        baseline_results[model_name] = baseline_model_results

    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Save both results
    optimized_output = args.output_file.replace('.json', '_optimized.json')
    baseline_output = args.output_file.replace('.json', '_baseline.json')
    
    with open(optimized_output, 'w') as f:
        json.dump(optimized_results, f, indent=4)
    with open(baseline_output, 'w') as f:
        json.dump(baseline_results, f, indent=4)

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