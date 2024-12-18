# abstention_rl_training.py

import os
import pickle
from collections import defaultdict
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pathlib

from data_utils.common_utils import ALL_OPTIONS
from data_utils import DATASETS

from uncertainty_quantification_via_cp import Abstention_CP

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(64, 3)
        self.log_std_head = nn.Linear(64, 3)

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        std = torch.exp(log_std)
        return mean, std

def train_rl_policy(args):
    # Initialize policy network and optimizer
    policy_net = PolicyNetwork()
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    models = os.listdir(args.result_data_path)

    for model_name in models:
        print(f"Training policy for model: {model_name}")
        model_path = os.path.join(args.result_data_path, model_name)
        
        # Create model-specific directory for saving policies
        model_save_path = pathlib.Path(args.save_model_path) / model_name
        model_save_path.mkdir(parents=True, exist_ok=True)

        # Initialize hyperparameters
        hyperparams = torch.tensor([args.alpha, args.beta, args.cum_prob_threshold])

        for dataset_name in DATASETS:
            print(f"Dataset: {dataset_name}")
            file_name = os.path.join(model_path, f"{dataset_name}.pkl")
            if not os.path.exists(file_name):
                continue
            with open(file_name, 'rb') as f:
                result_data = pickle.load(f)

            # Split data into calibration and test sets
            cal_result_data, test_result_data = train_test_split(
                result_data, train_size=args.cal_ratio, random_state=42
            )

            # Training loop
            for epoch in range(args.epochs):
                # Get current hyperparameters
                current_params = hyperparams

                # Get mean and std from policy network
                mean, std = policy_net(current_params)

                # Create a normal distribution and sample actions
                dist = torch.distributions.Normal(mean, std)
                actions = dist.sample()

                # Calculate log probabilities
                log_probs = dist.log_prob(actions)

                # Update hyperparameters
                new_params = current_params + actions * args.adjustment_scale
                # Ensure hyperparameters are within valid ranges
                alpha = torch.clamp(new_params[0], 0.001, 0.1)
                beta = torch.clamp(new_params[1], 0.001, 0.2)
                cum_prob_threshold = torch.clamp(new_params[2], 0.5, 1.0)

                # Update args for metrics calculation
                args.alpha = alpha.item()
                args.beta = beta.item()
                args.cum_prob_threshold = cum_prob_threshold.item()

                # Run environment step
                _, accuracy, abstention_rate, average_set_size = Abstention_CP(
                    cal_result_data, test_result_data, args
                )

                # Compute reward
                error_rate = 1.0 - accuracy
                cost = error_rate + args.lambda1 * average_set_size + args.lambda2 * abstention_rate
                reward = -cost  # Negative cost as reward

                # Compute policy loss
                loss = -log_probs.sum() * reward  # Multiply by reward

                # Backpropagate and update policy network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update hyperparameters for next iteration
                hyperparams = torch.tensor([alpha.item(), beta.item(), cum_prob_threshold.item()])

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Cost: {cost}, Hyperparameters: alpha={alpha.item()}, beta={beta.item()}, cum_prob_threshold={cum_prob_threshold.item()}")

            # Save the trained policy network with organized structure
            save_path = model_save_path / f"{model_name}_{dataset_name}_policy.pth"
            torch.save({
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'hyperparameters': {
                    'alpha': alpha.item(),
                    'beta': beta.item(),
                    'cum_prob_threshold': cum_prob_threshold.item()
                },
                'final_metrics': {
                    'accuracy': accuracy,
                    'abstention_rate': abstention_rate,
                    'average_set_size': average_set_size
                }
            }, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_data_path", type=str, required=True, help="Path to result data")
    parser.add_argument("--save_model_path", type=str, required=True, help="Path to save trained policies")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--adjustment_scale", type=float, default=0.05, help="Scale for hyperparameter adjustments")
    parser.add_argument("--lambda1", type=float, default=0.5, help="Weight for average set size in the cost function.")
    parser.add_argument("--lambda2", type=float, default=0.5, help="Weight for abstention rate in the cost function.")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Initial alpha value.")
    parser.add_argument("--beta", type=float, default=0.05,
                        help="Initial beta value.")
    parser.add_argument("--cum_prob_threshold", type=float, default=0.9,
                        help="Initial cumulative probability threshold.")
    parser.add_argument("--cal_ratio", type=float, default=0.5,
                        help="The ratio of data to be used as the calibration data.")
    args = parser.parse_args()

    os.makedirs(args.save_model_path, exist_ok=True)

    train_rl_policy(args)