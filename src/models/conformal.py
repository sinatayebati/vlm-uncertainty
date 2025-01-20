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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = args['alpha']
    beta = args['beta']

    cal_scores = []
    for row in cal_result_data:
        probs = softmax(row["logits"][:6])
        truth_answer = row["answer"]
        cal_scores.append(1 - probs[ALL_OPTIONS.index(truth_answer)])

    n = len(cal_result_data)
    q_level_predict = np.ceil((n + 1) * (1 - alpha)) / n
    q_level_abstain = np.ceil((n + 1) * (1 - beta)) / n

    q_level_predict = min(max(q_level_predict, 0.0), 1.0)
    q_level_abstain = min(max(q_level_abstain, 0.0), 1.0)

    qhat_predict = np.quantile(cal_scores, q_level_predict, method='higher')
    qhat_abstain = np.quantile(cal_scores, q_level_abstain, method='higher')

    qhat_predict = torch.tensor(qhat_predict, dtype=torch.float32, device=device)
    qhat_abstain = torch.tensor(qhat_abstain, dtype=torch.float32, device=device)

    pred_outputs = {}
    metrics = {
        "correct_predictions": 0.0,
        "total_predictions": 0.0,
        "abstentions": 0.0,
        "set_sizes": [],
        "total_instances": len(test_result_data),
    }

    for row in test_result_data:
        logits = torch.tensor(row["logits"][:6], dtype=torch.float32, device=device)
        probs = F.softmax(logits, dim=0)
        scores = [1 - prob for prob in probs]

        # Action probabilities for predicting single answer, predicting set, and abstain
        p_single = torch.sigmoid(-5 * (torch.min(torch.tensor(scores)) - qhat_predict))
        p_abstain = torch.sigmoid(5 * (torch.min(torch.tensor(scores)) - qhat_abstain))
        p_set = torch.clamp(1 - p_single - p_abstain, min=0.0)

        action_probs = torch.stack([p_single, p_set, p_abstain])
        action_probs = torch.clamp(action_probs, min=1e-6)

        if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
            action_probs = torch.tensor([0.4, 0.2, 0.4])
        else:
            action_probs = action_probs / action_probs.sum()

        # Sample action
        try:
            action = torch.multinomial(action_probs, 1).item()
        except RuntimeError:
            action = torch.argmax(action_probs).item()

        true_answer = row["answer"]

        if action == 0:
            predicted_label = ALL_OPTIONS[torch.argmax(probs).item()]
            pred_outputs[str(row["id"])] = {
                'prediction': predicted_label,
                'logits': row["logits"][:6]
            }
            if predicted_label == true_answer:
                metrics["correct_predictions"] += 1.0
            metrics["total_predictions"] += 1.0

        elif action == 1:
            probs = F.softmax(logits, dim=0)
            pred_set = []
            for idx, prob in enumerate(probs):
                if prob >= (1 - qhat_abstain):
                    pred_set.append(ALL_OPTIONS[idx])
            
            if len(pred_set) == 0:
                pred_set.append(ALL_OPTIONS[torch.argmax(probs).item()])
            
            pred_outputs[str(row["id"])] = {
                'prediction': pred_set,
                'logits': row["logits"][:6]
            }
            
            if true_answer in pred_set:
                metrics["correct_predictions"] += 1.0
            metrics["total_predictions"] += 1.0
            metrics["set_sizes"].append(len(pred_set))

        else:
            pred_outputs[str(row["id"])] = {
                'prediction': 'abstain',
                'logits': row["logits"][:6]
            }
            metrics["abstentions"] += 1.0

    # Compute metrics as tensors
    if metrics["total_predictions"] > 0:
        accuracy = torch.tensor(metrics["correct_predictions"] / metrics["total_predictions"], 
                              dtype=torch.float32, device=device)
    else:
        accuracy = torch.tensor(0.0, dtype=torch.float32)  # No predictions made

    abstention_rate = torch.tensor(metrics["abstentions"] / metrics["total_instances"], 
                                 dtype=torch.float32, device=device)
    if metrics["set_sizes"]:
        average_set_size = torch.tensor(sum(metrics["set_sizes"]) / len(metrics["set_sizes"]), 
                                       dtype=torch.float32, device=device)
    else:
        average_set_size = torch.tensor(0.0, dtype=torch.float32)

    return pred_outputs, accuracy, abstention_rate, average_set_size