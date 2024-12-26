from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
from data_utils.common_utils import ALL_OPTIONS

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def Abstention_CP(cal_result_data, test_result_data, args: Dict):
    """
    Differentiable version of Abstention_CP.
    """
    # Use the args directly (assumed to be floats)
    alpha = args['alpha']
    beta = args['beta']
    cum_prob_threshold = args['cum_prob_threshold']

    # Calculate thresholds using numpy
    n = len(cal_result_data)
    q_level_predict = np.ceil((n + 1) * (1 - alpha)) / n
    q_level_abstain = np.ceil((n + 1) * (1 - beta)) / n

    # Clamp q_level values to [0, 1]
    q_level_predict = min(max(q_level_predict, 0.0), 1.0)
    q_level_abstain = min(max(q_level_abstain, 0.0), 1.0)

    # Get calibration scores
    cal_scores = [1 - max(softmax(row["logits"][:6])) for row in cal_result_data]

    # Calculate qhat using numpy quantile
    qhat_predict = np.quantile(cal_scores, q_level_predict, method='higher')
    qhat_abstain = np.quantile(cal_scores, q_level_abstain, method='higher')

    # Convert qhat values to tensors
    qhat_predict = torch.tensor(qhat_predict, dtype=torch.float32)
    qhat_abstain = torch.tensor(qhat_abstain, dtype=torch.float32)

    pred_outputs = {}
    metrics = {
        "correct_predictions": 0.0,
        "total_predictions": 0.0,
        "abstentions": 0.0,
        "set_sizes": [],
        "total_instances": len(test_result_data),
    }

    for row in test_result_data:
        logits = torch.tensor(row["logits"][:6], dtype=torch.float32)
        probs = F.softmax(logits, dim=0)
        max_prob = torch.max(probs).item()
        score = torch.tensor(1 - max_prob, dtype=torch.float32)

        # Use smooth functions to compute action probabilities
        # Action probabilities for predicting single answer, predicting set, and abstain
        p_single = torch.sigmoid(-10 * (score - qhat_predict))
        p_abstain = torch.sigmoid(10 * (score - qhat_abstain))
        p_set = torch.clamp(1 - p_single - p_abstain, min=0.0)

        # Normalize probabilities
        action_probs = torch.stack([p_single, p_set, p_abstain])
        
        # Ensure no negative values (clip to small positive number)
        action_probs = torch.clamp(action_probs, min=1e-6)

        if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
            action_probs = torch.tensor([0.4, 0.2, 0.4])
        else:
            action_probs = action_probs / action_probs.sum()

        # Sample action
        try:
            action = torch.multinomial(action_probs, 1).item()
        except RuntimeError:
            # Fallback in case of sampling error
            action = torch.argmax(action_probs).item()

        true_answer = row["answer"]

        if action == 0:
            # Predict single answer
            predicted_label = ALL_OPTIONS[torch.argmax(probs).item()]
            pred_outputs[str(row["id"])] = {
                'prediction': predicted_label,
                'logits': row["logits"][:6]  # Store logits with prediction
            }
            if predicted_label == true_answer:
                metrics["correct_predictions"] += 1.0
            metrics["total_predictions"] += 1.0

        elif action == 1:
            # Predict set of answers
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            cum_threshold = torch.sigmoid(10 * (cumulative_probs - cum_prob_threshold))
            indices = torch.nonzero(cum_threshold >= 0.5, as_tuple=False)
            if indices.numel() > 0:
                set_size = indices[0].item() + 1
            else:
                set_size = len(cum_threshold)
            pred_set = [ALL_OPTIONS[idx.item()] for idx in sorted_indices[:set_size]]
            pred_outputs[str(row["id"])] = {
                'prediction': pred_set,
                'logits': row["logits"][:6]  # Store logits with prediction set
            }

            if true_answer in pred_set:
                metrics["correct_predictions"] += 1.0
            metrics["total_predictions"] += 1.0
            metrics["set_sizes"].append(set_size)

        else:
            # Abstain
            pred_outputs[str(row["id"])] = {
                'prediction': 'abstain',
                'logits': row["logits"][:6]  # Store logits even with abstention
            }
            metrics["abstentions"] += 1.0

    # Compute metrics as tensors
    if metrics["total_predictions"] > 0:
        accuracy = torch.tensor(metrics["correct_predictions"] / metrics["total_predictions"], dtype=torch.float32)
    else:
        accuracy = torch.tensor(0.0, dtype=torch.float32)  # No predictions made

    abstention_rate = torch.tensor(metrics["abstentions"] / metrics["total_instances"], dtype=torch.float32)
    if metrics["set_sizes"]:
        average_set_size = torch.tensor(sum(metrics["set_sizes"]) / len(metrics["set_sizes"]), dtype=torch.float32)
    else:
        average_set_size = torch.tensor(0.0, dtype=torch.float32)

    return pred_outputs, accuracy, abstention_rate, average_set_size