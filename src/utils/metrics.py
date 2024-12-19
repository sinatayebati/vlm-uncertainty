from typing import Dict, List, Union, Tuple
import numpy as np
from collections import Counter
from torchmetrics.classification import MulticlassCalibrationError
import torch

from data_utils.common_utils import ALL_OPTIONS

MAPPING = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5}

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_base_metrics(test_result_data: List[Dict]) -> Tuple[float, float, float]:
    """Compute accuracy and E/F ratios."""
    res = []
    preds = []
    for row in test_result_data:
        truth_answer = row["answer"]
        logits = row["logits"][:6]
        pred_answer = ALL_OPTIONS[np.argmax(logits)]
        preds.append(pred_answer)
        res.append(1 if pred_answer == truth_answer else 0)
    
    counts = Counter(preds)
    E_ratio = counts.get("E", 0) / len(preds)
    F_ratio = counts.get("F", 0) / len(preds)
    
    return E_ratio, F_ratio

def compute_calibration_error(result_data: List[Dict], norm: str = 'l1') -> float:
    """Compute calibration error (ECE or MCE based on norm)."""
    target = torch.tensor([MAPPING[row['answer']] for row in result_data])
    pred = torch.tensor(np.array([softmax(row['logits'][:6]) for row in result_data]))
    metric = MulticlassCalibrationError(num_classes=6, n_bins=15, norm=norm)
    return metric(pred, target).item()

def compute_prediction_distribution(pred_outputs: Dict) -> Dict[str, float]:
    """Compute distribution of prediction types (single, set, abstain)."""
    total = len(pred_outputs)
    singles = 0
    sets = 0
    abstains = 0
    
    for prediction in pred_outputs.values():
        if prediction == 'abstain':
            abstains += 1
        elif isinstance(prediction, list):
            sets += 1
        else:  # single prediction
            singles += 1
    
    return {
        'single_pred_rate': singles / total if total > 0 else 0.0,
        'set_pred_rate': sets / total if total > 0 else 0.0,
        'abstain_rate': abstains / total if total > 0 else 0.0
    }

def cal_set_size(pred_outputs):
    """Calculate average size of prediction sets (excluding single predictions and abstentions)."""
    sz = []
    for k, v in pred_outputs.items():
        sz.append(len(v))
    return sum(sz) /len(sz)

def compute_set_metrics(pred_outputs: Dict, test_id_to_answer: Dict) -> Dict[str, float]:
    """Compute metrics for prediction sets including coverage and set sizes."""
    correct_predictions = 0
    total_predictions = 0
    abstentions = 0
    coverage_count = 0

    for idx, prediction in pred_outputs.items():
        true_answer = test_id_to_answer[idx]
        
        if prediction == 'abstain':
            abstentions += 1
        elif isinstance(prediction, list):
            if true_answer in prediction:
                coverage_count += 1
                correct_predictions += 1
            total_predictions += 1
        else:  # single prediction
            if prediction == true_answer:
                coverage_count += 1
                correct_predictions += 1
            total_predictions += 1

    total_instances = len(pred_outputs)
    pred_distribution = compute_prediction_distribution(pred_outputs)
    set_sizes = cal_set_size(pred_outputs)

    metrics = {
        'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0,
        'coverage': coverage_count / total_instances,
        'set_sizes': set_sizes,
        'uacc': (correct_predictions / total_predictions) * np.sqrt(len(ALL_OPTIONS)) / np.mean(set_sizes) if set_sizes else 0.0,
        **pred_distribution
    }
    
    return metrics
