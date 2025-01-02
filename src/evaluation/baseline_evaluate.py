import pickle
import json
from typing import Literal, Dict, List, Union, Tuple
import os
from collections import Counter, defaultdict
import argparse
import math

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError

from data_utils.common_utils import ALL_OPTIONS
from data_utils import DATASETS, SEEDBENCH_CATS, OODCV_CATS

MAPPING = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_accuracy(test_result_data):
    res = []
    preds = []
    for row in test_result_data:
        truth_answer = row["answer"]
        logits = row["logits"][:6]
        pred_answer = ALL_OPTIONS[np.argmax(logits)]
        preds.append(pred_answer)
        if pred_answer == truth_answer:
            res.append(1)
        else:
            res.append(0)
    return sum(res) / len(res), preds


def get_ce(result_data, norm):
    target = torch.tensor([MAPPING[row['answer']] for row in result_data])
    pred = torch.tensor(
        np.array(
            [softmax(row['logits'][:6]) for row in result_data]
        )
    )
    metric = MulticlassCalibrationError(num_classes=6, n_bins=15, norm=norm)
    res = metric(pred, target)
    return res.item()


def cal_acc(test_result_data):
    acc, preds = get_accuracy(test_result_data)
    counts = Counter(preds)
    E_ratio = counts["E"] / len(preds)
    F_ratio = counts["F"] / len(preds)
    return acc, E_ratio, F_ratio

def LAC_CP(cal_result_data, test_result_data, alpha=0.1):
    """
    Apply conformal prediction using LAC score function.
    Now includes logits with predictions.
    """
    cal_scores = []
    for row in cal_result_data:
        probs = softmax(row["logits"][:6])
        truth_answer = row["answer"]
        cal_scores.append(1 - probs[ALL_OPTIONS.index(truth_answer)])

    n = len(cal_result_data)
    q_level = np.ceil((n+1) * (1-alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    
    pred_sets = {}
    for row in test_result_data:
        probs = softmax(row["logits"][:6])
        ps = []
        for ii, p in enumerate(probs[:6]):
            if p >= 1 - qhat:
                ps.append(ALL_OPTIONS[ii])
        if len(ps) == 0:
            ps.append(ALL_OPTIONS[np.argmax(probs)])
        
        pred_sets[str(row["id"])] = {
            'prediction': ps,
            'logits': row["logits"][:6]
        }
    return pred_sets

def APS_CP(cal_result_data, test_result_data, alpha=0.1):
    """
    Apply conformal prediction using APS score function.
    Now includes logits with predictions.
    """
    cal_scores = []
    for row in cal_result_data:
        probs = softmax(row["logits"][:6])
        truth_answer = row["answer"]
        cal_pi = np.argsort(probs)[::-1]
        cal_sum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()
        cal_sum_r = np.take_along_axis(cal_sum, cal_pi.argsort(), axis=0)
        cal_score = cal_sum_r[ALL_OPTIONS.index(truth_answer)]
        cal_scores.append(cal_score)

    n = len(cal_result_data)
    q_level = np.ceil((n+1) * (1-alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    
    pred_sets = {}
    for row in test_result_data:
        probs = softmax(row["logits"][:6])
        cal_pi = np.argsort(probs)[::-1]
        cal_sum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()
        ps = []
        ii = 0
        while ii < min(len(cal_sum), 6) and cal_sum[ii] <= qhat:
            op_id = cal_pi[ii]
            ps.append(ALL_OPTIONS[op_id])
            ii += 1
        if len(ps) == 0:
            op_id = cal_pi[ii]
            ps.append(ALL_OPTIONS[op_id])
            
        pred_sets[str(row["id"])] = {
            'prediction': ps,
            'logits': row["logits"][:6]
        }
    return pred_sets


def compute_baseline_auroc(pred_sets: Dict, test_id_to_answer: Dict) -> float:
    """Compute AUROC for baseline methods."""
    confidences = []
    correctness = []
    
    for idx, output in pred_sets.items():
        true_answer = test_id_to_answer[idx]
        prediction = output['prediction']
        logits = output['logits']
        probs = softmax(logits)
        
        # For set predictions, use maximum probability among predicted classes
        pred_probs = [probs[ALL_OPTIONS.index(p)] for p in prediction]
        confidence = max(pred_probs)
        correct = true_answer in prediction
            
        confidences.append(confidence)
        correctness.append(1.0 if correct else 0.0)
    
    if len(confidences) == 0 or len(set(confidences)) < 2 or len(set(correctness)) < 2:
        return 0.5
        
    try:
        return roc_auc_score(correctness, confidences)
    except ValueError:
        return 0.5


def compute_baseline_auarc(pred_outputs: Dict, test_id_to_answer: Dict) -> float:
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

        pred_probs = [probs[MAPPING[p]] for p in prediction]
        confidence = max(pred_probs)
        correct = true_answer in prediction
            
        confidences.append(confidence)
        correctness.append(1.0 if correct else 0.0)

    confidences = np.array(confidences)
    correctness = np.array(correctness)
    
    sort_idx = np.argsort(confidences)
    sorted_correctness = correctness[sort_idx]
    
    mean_accuracies = np.cumsum(sorted_correctness[::-1]) / np.arange(1, len(sorted_correctness) + 1)
    
    coverage_points = np.linspace(0, 1, len(sorted_correctness))
    
    return auc(coverage_points, mean_accuracies)


def cal_coverage(pred_sets, test_id_to_answer):
    """
    Calculate the coverage rate of prediction sets.
    """
    cover = []
    for k, v in pred_sets.items():
        if test_id_to_answer[k] in v['prediction']:
            cover.append(1)
        else:
            cover.append(0)
    return sum(cover) / len(cover)

def cal_set_size(pred_sets):
    sz = []
    for k, v in pred_sets.items():
        sz.append(len(v['prediction']))
    return sum(sz) /len(sz)


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


def calculate_metrics(result_data, args):
    cal_result_data, test_result_data = train_test_split(
        result_data, train_size=args.cal_ratio, random_state=42
    )

    test_id_to_answer = {str(row["id"]): row["answer"] for row in test_result_data}

    acc, E_ratio, F_ratio = cal_acc(test_result_data)
    
    # Get prediction sets from both methods
    pred_sets_LAC = LAC_CP(cal_result_data, test_result_data, alpha=args.alpha)
    pred_sets_APS = APS_CP(cal_result_data, test_result_data, alpha=args.alpha)
    
    # Compute metrics for LAC
    coverage_LAC = cal_coverage(pred_sets_LAC, test_id_to_answer)
    set_sizes_LAC = cal_set_size(pred_sets_LAC)
    auroc_LAC = compute_baseline_auroc(pred_sets_LAC, test_id_to_answer)
    auarc_LAC = compute_baseline_auarc(pred_sets_LAC, test_id_to_answer)
    uacc_LAC = acc * np.sqrt(len(ALL_OPTIONS)) / set_sizes_LAC

    # Compute metrics for APS
    coverage_APS = cal_coverage(pred_sets_APS, test_id_to_answer)
    set_sizes_APS = cal_set_size(pred_sets_APS)
    auroc_APS = compute_baseline_auroc(pred_sets_APS, test_id_to_answer)
    auarc_APS = compute_baseline_auarc(pred_sets_APS, test_id_to_answer)
    uacc_APS = acc * np.sqrt(len(ALL_OPTIONS)) / set_sizes_APS

    # Calculate ECE and MCE
    ece = get_ce(result_data=result_data, norm='l1')
    mce = get_ce(result_data=result_data, norm='max')

    return {
        "E_ratio": E_ratio,
        "F_ratio": F_ratio,
        "coverage_LAC": coverage_LAC,
        "set_sizes_LAC": set_sizes_LAC,
        "uacc_LAC": uacc_LAC,
        "auroc_LAC": auroc_LAC,
        "auarc_LAC": auarc_LAC,
        "coverage_APS": coverage_APS,
        "set_sizes_APS": set_sizes_APS,
        "uacc_APS": uacc_APS,
        "auroc_APS": auroc_APS,
        "auarc_APS": auarc_APS,
        "ece": ece,
        "mce": mce,
        "accuracy": acc,
        "coverage": np.mean([coverage_LAC, coverage_APS]),
        "set_sizes": np.mean([set_sizes_LAC, set_sizes_APS]),
        "uacc": np.mean([uacc_LAC, uacc_APS])
    }

def calculate_metrics_for_model(model_name, args):
    """Modified to store results by dataset"""
    model_results = {}
    for dataset_name in DATASETS:
        file_name = f'{args.result_data_path}/{model_name}/{dataset_name}.pkl'
        with open(file_name, 'rb') as f:
            result_data = pickle.load(f)
        
        model_results[dataset_name] = calculate_metrics(result_data, args)
    return model_results

def calculate_metrics_for_seedbench(model_name, args):
    """Modified to store results by category"""
    seedbench_results = {}
    file_name = f'{args.result_data_path}/{model_name}/seedbench.pkl'
    with open(file_name, 'rb') as f:
        result_data = pickle.load(f)
        for i in range(1, 10):
            cat_data = [row for row in result_data if row['question_type_id'] == i]
            print(f"Category {i}, length: ", len(cat_data))
            seedbench_results[f"category_{i}"] = calculate_metrics(cat_data, args)
    return seedbench_results

def calculate_metrics_for_oodcv(model_name, args):
    """Modified to store results by situation"""
    oodcv_results = {}
    file_name = f'{args.result_data_path}/{model_name}/oodcv.pkl'
    with open(file_name, 'rb') as f:
        result_data = pickle.load(f)
        for situation in OODCV_CATS:
            cat_data = [row for row in result_data if row['situation'] == situation]
            oodcv_results[situation] = calculate_metrics(cat_data, args)
    return oodcv_results

def main(args):
    full_result = {}
    model_names = os.listdir(args.result_data_path)
    
    for model_name in tqdm(model_names):
        if args.mode == "all":
            model_metrics = calculate_metrics_for_model(model_name, args)
        elif args.mode == "seedbench":
            model_metrics = calculate_metrics_for_seedbench(model_name, args)
        elif args.mode == "oodcv":
            model_metrics = calculate_metrics_for_oodcv(model_name, args)
        else:
            raise ValueError(f"Unrecognized mode: {args.mode}")
        
        full_result[model_name] = model_metrics

    with open(args.file_to_write, 'w') as f:
        json.dump(full_result, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_data_path", type=str, required=True)
    parser.add_argument("--file_to_write", type=str, required=True)
    parser.add_argument("--mode", choices=["all", "seedbench", "oodcv"], default="all")
    parser.add_argument("--cal_ratio", type=float, default=0.5,
                        help="The ratio of data to be used as the calibration data.")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="The error rate parameter for predictions.")
    args = parser.parse_args()

    main(args)
    #python -m uncertainty_quantification_via_cp --result_data_path 'output' --file_to_write 'full_result.json'
    #python -m uncertainty_quantification_via_cp --result_data_path 'output' --mode 'seedbench' --file_to_write 'full_result_seedbench.json'
    #python -m uncertainty_quantification_via_cp --result_data_path 'output' --mode 'oodcv' --file_to_write 'full_result_oodcv.json'