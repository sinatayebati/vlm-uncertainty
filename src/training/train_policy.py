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

def calculate_prediction_diversity_bonus(single_pred_count, set_pred_count, abstain_count, total_count):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = torch.tensor([
        single_pred_count/total_count,
        set_pred_count/total_count,
        abstain_count/total_count
    ], device=device)
    
    eps = 1e-8
    entropy = -torch.sum(props * torch.log(props + eps))
    return entropy

def train_rl_policy(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy_net = PolicyNetwork().to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=config['learning_rate'])
    models = os.listdir(args.result_data_path)

    for model_name in models:
        print(f"Training policy for model: {model_name}")
        model_path = os.path.join(args.result_data_path, model_name)
        
        model_save_path = pathlib.Path(args.save_model_path) / model_name
        model_save_path.mkdir(parents=True, exist_ok=True)

        hyperparams = torch.tensor([config['alpha'], config['beta']], device=device)

        for dataset_name in DATASETS:
            print(f"Dataset: {dataset_name}")
            file_name = os.path.join(model_path, f"{dataset_name}.pkl")
            if not os.path.exists(file_name):
                continue
            with open(file_name, 'rb') as f:
                result_data = pickle.load(f)

            cal_result_data, test_result_data = train_test_split(
                result_data, train_size=config['cal_ratio'], random_state=42
            )

            # Training loop
            for epoch in range(config['epochs']):
                current_params = hyperparams.to(device)

                mean, std = policy_net(current_params)

                dist = torch.distributions.Normal(mean, std)
                actions = dist.sample()

                log_probs = dist.log_prob(actions)

                new_params = current_params + actions * config['adjustment_scale']
                
                alpha = torch.clamp(new_params[0], 0.75, 0.9)
                beta = torch.clamp(new_params[1], 0.01, 0.05)

                config['alpha'] = alpha.item()
                config['beta'] = beta.item()

                pred_outputs, accuracy, abstention_rate, average_set_size = Abstention_CP(
                    cal_result_data, test_result_data, config
                )

                single_pred_count = sum(1 for p in pred_outputs.values() if isinstance(p['prediction'], str) and p['prediction'] != 'abstain')
                set_pred_count = sum(1 for p in pred_outputs.values() if isinstance(p['prediction'], list))
                abstain_count = sum(1 for p in pred_outputs.values() if p['prediction'] == 'abstain')
                total_count = len(pred_outputs)

                single_pred_ratio = single_pred_count / total_count
                set_pred_ratio = set_pred_count / total_count
                abstain_ratio = abstain_count / total_count

                coverage = 1.0 - abstention_rate

                diversity_bonus = calculate_prediction_diversity_bonus(
                    single_pred_count, set_pred_count, abstain_count, total_count
                )

                error_rate = 1.0 - accuracy
                base_cost = error_rate + config['lambda1'] * average_set_size + config['lambda2'] * abstention_rate
                cost = base_cost - config['lambda3'] * coverage - config['lambda4'] * diversity_bonus

                if single_pred_ratio > 0.8:
                    cost += config['lambda5'] * (single_pred_ratio - 0.8)

                reward = -cost

                loss = -log_probs.sum() * reward

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                hyperparams = torch.tensor([alpha.item(), beta.item()], device=device)

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Cost: {cost}, Hyperparameters: alpha={alpha.item()}, beta={beta.item()}, cum_prob_threshold={cum_prob_threshold.item()}")

            save_path = model_save_path / f"{model_name}_{dataset_name}_policy.pth"
            torch.save({
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'hyperparameters': {
                    'alpha': alpha.item(),
                    'beta': beta.item(),
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

    config = load_config(args.config_file)

    os.makedirs(args.save_model_path, exist_ok=True)

    train_rl_policy(args, config)

def run_module():
    main()

if __name__ == "__main__":
    run_module()