from typing import Dict, List, Union, Tuple
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score, auc
from torchmetrics.classification import MulticlassCalibrationError
import torch

from data_utils.common_utils import ALL_OPTIONS

MAPPING = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5}

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def compute_auroc(pred_outputs: Dict, test_id_to_answer: Dict) -> float:
    """
    Compute Area Under ROC curve for prediction confidence vs correctness.
    """
    confidences = []
    correctness = []
    
    for idx, output in pred_outputs.items():
        true_answer = test_id_to_answer[idx]
        prediction = output['prediction']  
        logits = output['logits']
        probs = softmax(logits[:6])  
        
        if prediction == 'abstain':
            confidence = 1.0
            correct = True
        elif isinstance(prediction, list):
            true_answer_prob = probs[MAPPING[true_answer]] if true_answer in prediction else 0
            pred_probs = [probs[MAPPING[p]] for p in prediction]
            max_prob = max(pred_probs)
            
            confidence = 0.7 * max_prob + 0.3 * true_answer_prob
            correct = true_answer in prediction
        else:
            confidence = probs[MAPPING[prediction]]
            correct = prediction == true_answer
            
        confidences.append(confidence)
        correctness.append(1.0 if correct else 0.0)
    
    if len(confidences) == 0 or len(set(confidences)) < 2 or len(set(correctness)) < 2:
        return 0.5
        
    try:
        return roc_auc_score(correctness, confidences)
    except ValueError:
        return 0.5

def compute_auarc(pred_outputs: Dict, test_id_to_answer: Dict) -> float:
    """
    Compute Area Under Accuracy-Rejection Curve (AUARC).
    """

    confidences = []
    correctness = []
    
    for idx, output in pred_outputs.items():
        true_answer = test_id_to_answer[idx]
        prediction = output['prediction']
        logits = output['logits']
        probs = softmax(logits[:6])

        if prediction == 'abstain':
            confidence = 1.0
            correct = True
        elif isinstance(prediction, list):
            pred_probs = [probs[MAPPING[p]] for p in prediction]
            confidence = max(pred_probs)
            correct = true_answer in prediction
        else:
            confidence = probs[MAPPING[prediction]]
            correct = prediction == true_answer
            
        confidences.append(confidence)
        correctness.append(1.0 if correct else 0.0)

    confidences = np.array(confidences)
    correctness = np.array(correctness)
    
    sort_idx = np.argsort(confidences)
    sorted_correctness = correctness[sort_idx]
    
    mean_accuracies = np.cumsum(sorted_correctness[::-1]) / np.arange(1, len(sorted_correctness) + 1)
    
    coverage_points = np.linspace(0, 1, len(sorted_correctness))
    
    return auc(coverage_points, mean_accuracies)


def compute_calibration_error(pred_outputs: Dict[str, Dict], test_id_to_answer: Dict[str, str],n_bins: int = 15) -> float:
    """
    Compute calibration error
    """
    confidences = []
    correctnesses = []

    for idx, output in pred_outputs.items():
        true_answer = test_id_to_answer[idx]
        prediction = output['prediction']
        logits = output['logits'][:6]
        probs = softmax(logits)

        if prediction == 'abstain':
            continue

        elif isinstance(prediction, str):
            conf = probs[MAPPING[prediction]]
            correct = 1.0 if (prediction == true_answer) else 0.0
            confidences.append(conf)
            correctnesses.append(correct)

        else:
            set_prob = sum(probs[MAPPING[p]] for p in prediction)
            conf = set_prob
            correct = 1.0 if (true_answer in prediction) else 0.0
            confidences.append(conf)
            correctnesses.append(correct)

    if not confidences:
        return 0.0

    confidences = np.array(confidences)
    correctnesses = np.array(correctnesses)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries, right=True)

    total_samples = len(confidences)
    ece = 0.0

    for b in range(1, n_bins + 1):
        bin_mask = (bin_indices == b)
        bin_count = np.sum(bin_mask)
        if bin_count == 0:
            continue

        avg_conf = np.mean(confidences[bin_mask])
        avg_corr = np.mean(correctnesses[bin_mask])
        ece_bin = abs(avg_conf - avg_corr) * (bin_count / total_samples)
        ece += ece_bin

    return ece


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


def compute_prediction_distribution(pred_outputs: Dict) -> Dict[str, float]:
    """Compute distribution of prediction types (single, set, abstain)."""
    total = len(pred_outputs)
    singles = 0
    sets = 0
    abstains = 0
    
    for output in pred_outputs.values():
        prediction = output['prediction']  
        if prediction == 'abstain':
            abstains += 1
        elif isinstance(prediction, list):
            sets += 1
        else:
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
        prediction = v['prediction'] 
        sz.append(len(prediction))
    return sum(sz) / len(sz) if sz else 0.0


def compute_precision_at_k(pred_outputs: Dict, test_id_to_answer: Dict, k: int) -> float:
    """
    Compute precision@k: accuracy of predictions when prediction set size is exactly k.
    Returns (precision@k, coverage@k) tuple.
    """
    correct_at_k = []
    cover_all = []
    total_predictions = len(pred_outputs)
    
    for idx, output in pred_outputs.items():
        prediction = output['prediction']
        true_answer = test_id_to_answer[idx]
        
        if prediction == 'abstain':
            continue
        
        pred_set = [prediction] if isinstance(prediction, str) else prediction
        
        if len(pred_set) == k:
            if true_answer in pred_set:
                cover_all.append(1)
                correct_at_k.append(1)
            else:
                cover_all.append(0)
                correct_at_k.append(0)
            
    precision = sum(correct_at_k) / len(correct_at_k) if correct_at_k else 0.0
    coverage = sum(cover_all) / total_predictions if total_predictions > 0 else 0.0
    
    return precision, coverage


def compute_accuracy_vs_size_curve(pred_outputs: Dict, test_id_to_answer: Dict) -> Dict[str, List[float]]:
    """
    Compute accuracy and coverage for each prediction set size.
    Returns dictionary with accuracies, coverages, and sizes.
    """
    max_size = max(len(output['prediction']) if isinstance(output['prediction'], list) else 1 
                   for output in pred_outputs.values())
    
    accuracies = []
    coverages = []
    sizes = []
    
    for k in range(1, max_size + 1):
        precision, coverage = compute_precision_at_k(pred_outputs, test_id_to_answer, k)
        accuracies.append(precision)
        coverages.append(coverage)
        sizes.append(k)
    
    return {
        'accuracies': accuracies,
        'coverages': coverages,
        'sizes': sizes
    }


def compute_set_metrics(pred_outputs: Dict, test_id_to_answer: Dict) -> Dict[str, float]:
    """Compute metrics for prediction sets including coverage and set sizes."""
    cover = []
    correct_predictions = []

    for idx, output in pred_outputs.items():
        true_answer = test_id_to_answer[idx]
        prediction = output['prediction']
        logits = output['logits'][:6]  
        probs = softmax(logits)  
        
        if prediction == 'abstain':
            correct_predictions.append(1)
            cover.append(1)
            continue
        elif isinstance(prediction, list):
            pred_probs = [(p, probs[MAPPING[p]]) for p in prediction]
            max_prob_pred = max(pred_probs, key=lambda x: x[1])[0]
            
            if max_prob_pred == true_answer:
                correct_predictions.append(1)
            elif true_answer in prediction:
                correct_predictions.append(1 / len(prediction))
            else:
                correct_predictions.append(0)
                
            if true_answer in prediction:
                cover.append(1)
            else:
                cover.append(0)
        else:
            if prediction == true_answer:
                cover.append(1)
                correct_predictions.append(1)
            else:
                cover.append(0)
                correct_predictions.append(0)

    total_instance = len(pred_outputs)
    pred_distribution = compute_prediction_distribution(pred_outputs)
    set_sizes = cal_set_size(pred_outputs)
    auroc = compute_auroc(pred_outputs, test_id_to_answer)
    auarc = compute_auarc(pred_outputs, test_id_to_answer)

    metrics = {
        'accuracy': sum(correct_predictions) / len(correct_predictions) if correct_predictions else 0.0,
        'coverage': sum(cover) / total_instance if cover else 0.0,
        'set_sizes': set_sizes,
        'uacc': (sum(correct_predictions) / len(correct_predictions)) * np.sqrt(len(ALL_OPTIONS)) / set_sizes if set_sizes and correct_predictions else 0.0,
        'auroc': auroc,
        'auarc': auarc,
        **pred_distribution
    }
    
    return metrics