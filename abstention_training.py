import argparse
import json
import numpy as np
from collections import defaultdict
from skopt import gp_minimize
from skopt.space import Real
import pickle
import os

from uncertainty_quantification_via_cp import calculate_metrics
from data_utils import DATASETS

def calculate_metrics_for_model(model_name, args):
    model_results = defaultdict(list)
    model_dir = os.path.join(args.result_data_path, model_name)
    for dataset_name in DATASETS:
        file_name = os.path.join(model_dir, f"{dataset_name}.pkl")
        if not os.path.exists(file_name):
            print(f"File not found: {file_name}")
            continue
        with open(file_name, 'rb') as f:
            result_data = pickle.load(f)
            model_results = calculate_metrics(result_data, args, model_results)
    return model_results

def objective(params, args, model_name):
    alpha, beta, cum_prob_threshold, lambda1, lambda2 = params
    # Update args with the current hyperparameters
    args.alpha = alpha
    args.beta = beta
    args.cum_prob_threshold = cum_prob_threshold
    args.lambda1 = lambda1
    args.lambda2 = lambda2

    # Calculate metrics for the model over all datasets
    model_results = calculate_metrics_for_model(model_name, args)
    
    # Average the metrics over all datasets
    error_rates = [1 - acc if acc is not None else 1.0 for acc in model_results['accuracy']]
    average_set_sizes = model_results['average_set_size']
    abstention_rates = model_results['abstention_rate']

    # Handle empty metrics (if no data was processed)
    if not error_rates or not average_set_sizes or not abstention_rates:
        print(f"No data processed for model {model_name}. Skipping.")
        return np.inf  # Return a high cost to indicate failure

    # Aggregate the metrics
    error_rate = np.mean(error_rates)
    average_set_size = np.mean(average_set_sizes)
    abstention_rate = np.mean(abstention_rates)

    # Compute cost
    cost = error_rate + lambda1 * average_set_size + lambda2 * abstention_rate

    # For tracking purposes
    print(f"Model: {model_name}, alpha: {alpha:.4f}, beta: {beta:.4f}, cum_prob_threshold: {cum_prob_threshold:.4f}, "
          f"lambda1: {lambda1:.4f}, lambda2: {lambda2:.4f} => Cost: {cost:.4f}")

    return cost

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_data_path", type=str, required=True,
                        help="Path to the directory containing model result directories.")
    parser.add_argument("--file_to_write", type=str, required=True,
                        help="Path to the output JSON file for storing optimization results.")
    parser.add_argument("--cal_ratio", type=float, default=0.5,
                        help="The ratio of data to be used as the calibration data.")
    parser.add_argument("--n_calls", type=int, default=50, help="Number of optimization iterations.")
    parser.add_argument("--early_stopping_rounds", type=int, default=10,
                        help="Number of rounds to wait for improvement before early stopping.")
    args = parser.parse_args()

    # Get the list of model names
    model_names = os.listdir(args.result_data_path)

    # We will store the best hyperparameters for each model
    all_model_results = {}

    for model_name in model_names:
        model_dir = os.path.join(args.result_data_path, model_name)
        if not os.path.isdir(model_dir):
            continue  # Skip if not a directory
        print(f"\nOptimizing for model: {model_name}")

        # Define the search space for hyperparameters
        space = [
            Real(0.01, 0.5, name='alpha'),                # alpha between 0.01 and 0.5
            Real(0.01, 0.5, name='beta'),                 # beta between 0.01 and 0.5
            Real(0.5, 1.0, name='cum_prob_threshold'),    # cumulative probability threshold between 0.5 and 1.0
            Real(0.0, 1.0, name='lambda1'),               # lambda1 between 0 and 1
            Real(0.0, 1.0, name='lambda2'),               # lambda2 between 0 and 1
        ]

        # Keep track of the evaluation history
        evaluation_history = []

        # Define a wrapper for the objective function to record the history
        def objective_with_history(params):
            cost = objective(params, args, model_name)
            evaluation_history.append((params, cost))
            return cost

        # Run Bayesian Optimization
        res = gp_minimize(
            func=objective_with_history,
            dimensions=space,
            acq_func='EI',        # Expected Improvement.
            n_calls=args.n_calls,
            n_initial_points=10,
            random_state=42,
        )

        # Extract the best hyperparameters
        best_params = res.x
        best_cost = res.fun
        print(f"Best Parameters for model {model_name}:")
        print(f"alpha: {best_params[0]:.4f}, beta: {best_params[1]:.4f}, cum_prob_threshold: {best_params[2]:.4f}, "
              f"lambda1: {best_params[3]:.4f}, lambda2: {best_params[4]:.4f} => Cost: {best_cost:.4f}")

        # Save the evaluation history and best parameters
        results_to_save = {
            'best_params': {
                'alpha': best_params[0],
                'beta': best_params[1],
                'cum_prob_threshold': best_params[2],
                'lambda1': best_params[3],
                'lambda2': best_params[4],
                'cost': best_cost,
            },
            'evaluation_history': [
                {
                    'params': {
                        'alpha': params[0],
                        'beta': params[1],
                        'cum_prob_threshold': params[2],
                        'lambda1': params[3],
                        'lambda2': params[4],
                    },
                    'cost': cost
                } for params, cost in evaluation_history
            ]
        }

        # Store results for all models
        all_model_results[model_name] = results_to_save

    # Save all results to file
    with open(args.file_to_write, 'w') as f:
        json.dump(all_model_results, f, indent=4)

if __name__ == "__main__":
    main()