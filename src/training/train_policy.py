import os
import argparse
import yaml
from pathlib import Path
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pathlib
import pickle

from src.models.policy_network import PolicyNetwork
from src.models.conformal import Abstention_CP
from data_utils import DATASETS

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_rl_policy(args, config):
    # Initialize policy network and optimizer
    policy_net = PolicyNetwork()
    optimizer = optim.Adam(policy_net.parameters(), lr=config['learning_rate'])
    models = os.listdir(args.result_data_path)

    for model_name in models:
        print(f"Training policy for model: {model_name}")
        model_path = os.path.join(args.result_data_path, model_name)
        
        # Create model-specific directory for saving policies
        model_save_path = pathlib.Path(args.save_model_path) / model_name
        model_save_path.mkdir(parents=True, exist_ok=True)

        # Initialize hyperparameters
        hyperparams = torch.tensor([config['alpha'], config['beta'], config['cum_prob_threshold']])

        for dataset_name in DATASETS:
            print(f"Dataset: {dataset_name}")
            file_name = os.path.join(model_path, f"{dataset_name}.pkl")
            if not os.path.exists(file_name):
                continue
            with open(file_name, 'rb') as f:
                result_data = pickle.load(f)

            # Split data into calibration and test sets
            cal_result_data, test_result_data = train_test_split(
                result_data, train_size=config['cal_ratio'], random_state=42
            )

            # Training loop
            for epoch in range(config['epochs']):
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
                new_params = current_params + actions * config['adjustment_scale']
                # Ensure hyperparameters are within valid ranges
                alpha = torch.clamp(new_params[0], 0.001, 0.1)
                beta = torch.clamp(new_params[1], 0.001, 0.2)
                cum_prob_threshold = torch.clamp(new_params[2], 0.5, 1.0)

                # Update args for metrics calculation
                config['alpha'] = alpha.item()
                config['beta'] = beta.item()
                config['cum_prob_threshold'] = cum_prob_threshold.item()

                # Run environment step
                _, accuracy, abstention_rate, average_set_size = Abstention_CP(
                    cal_result_data, test_result_data, config
                )

                # Compute reward
                error_rate = 1.0 - accuracy
                cost = error_rate + config['lambda1'] * average_set_size + config['lambda2'] * abstention_rate
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


def main():
    parser = argparse.ArgumentParser(description="Train policy network using RL")
    parser.add_argument("--result_data_path", type=str, required=True,
                        help="Path to result data")
    parser.add_argument("--save_model_path", type=str, required=True,
                        help="Path to save trained policies")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_file)

    # Create save directory if it doesn't exist
    os.makedirs(args.save_model_path, exist_ok=True)

    # Run training
    train_rl_policy(args, config)

# Use this pattern for module execution
def run_module():
    main()

if __name__ == "__main__":
    run_module()