import pickle
import json
from typing import Literal
import os
from collections import Counter, defaultdict
import argparse
import math

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError

from data_utils.common_utils import ALL_OPTIONS
from data_utils import DATASETS, SEEDBENCH_CATS, OODCV_CATS

MAPPING = {'A':0 , 'B':1, 'C':2, 'D':3}

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
    Apply conformal prediction to obtain sets of predicted answers on each instance based on its softmax scores.
    Here the LAC score function is utilized.
    """
    cal_scores = []

    for row in cal_result_data:
        probs = softmax(row["logits"][:6])
        truth_answer = row["answer"]
        cal_scores.append(1 - probs[ALL_OPTIONS.index(truth_answer)])
    # calculate the threshold qhat
    n = len(cal_result_data)
    q_level = np.ceil((n+1) * (1-alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    # print(f"{m}_{fs} quantile: {qhat}")
    # generate prediction sets
    pred_sets = {}
    for row in test_result_data:
        probs = softmax(row["logits"][:6])
        ps = []
        for ii, p in enumerate(probs[:6]):
            # 1 - p <= qhat, so p >= 1- qhat
            if p >= 1 - qhat:
                ps.append(ALL_OPTIONS[ii])
        if len(ps) == 0:
            ps.append(ALL_OPTIONS[np.argmax(probs)])
        pred_sets[str(row["id"])] = ps
    return pred_sets


def APS_CP(cal_result_data, test_result_data, alpha=0.1):
    """
    Apply conformal prediction to obtain sets of predicted answers on each instance based on its softmax scores.
    Here the APS score function is utilized.
    """
    cal_scores = []
    for row in cal_result_data:
        probs = softmax(row["logits"][:6])
        truth_answer = row["answer"]
        cal_pi = np.argsort(probs)[::-1] # descending order
        cal_sum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()
        cal_sum_r = np.take_along_axis(cal_sum, cal_pi.argsort(), axis=0)
        cal_score = cal_sum_r[ALL_OPTIONS.index(truth_answer)]
        cal_scores.append(cal_score)
    n = len(cal_result_data)
    q_level = np.ceil((n+1) * (1-alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    # print(f"{m}_{fs} quantile: {qhat}")
    # generate prediction sets
    pred_sets = {}
    for row in test_result_data:
        probs = softmax(row["logits"][:6])
        cal_pi = np.argsort(probs)[::-1] # descending order
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
        pred_sets[str(row["id"])] = ps
    return pred_sets


def cal_coverage(pred_sets, test_id_to_answer):
    """
    Calculate the coverage rate of prediction sets.
    """""
    cover = []
    for k, v in pred_sets.items():
        if test_id_to_answer[k] in v:
            cover.append(1)
        else:
            cover.append(0)
    return sum(cover) / len(cover)

def cal_set_size(pred_sets):
    sz = []
    for k, v in pred_sets.items():
        sz.append(len(v))
    return sum(sz) /len(sz)



def Abstention_CP(cal_result_data, test_result_data, args):
    """
    Differentiable version of Abstention_CP.
    """
    # Use the args directly (assumed to be floats)
    alpha = args.alpha
    beta = args.beta
    cum_prob_threshold = args.cum_prob_threshold

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
            pred_outputs[str(row["id"])] = predicted_label
            if predicted_label == true_answer:
                metrics["correct_predictions"] += 1.0
            metrics["total_predictions"] += 1.0

        elif action == 1:
            # Predict set of answers
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            # Use smooth threshold for cumulative probability
            cum_threshold = torch.sigmoid(10 * (cumulative_probs - cum_prob_threshold))
            # Find indices where cum_threshold >= 0.5
            indices = torch.nonzero(cum_threshold >= 0.5, as_tuple=False)
            if indices.numel() > 0:
                set_size = indices[0].item() + 1
            else:
                set_size = len(cum_threshold)
            pred_set = [ALL_OPTIONS[idx.item()] for idx in sorted_indices[:set_size]]
            pred_outputs[str(row["id"])] = pred_set

            if true_answer in pred_set:
                metrics["correct_predictions"] += 1.0
            metrics["total_predictions"] += 1.0
            metrics["set_sizes"].append(set_size)

        else:
            # Abstain
            pred_outputs[str(row["id"])] = 'abstain'
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



def calculate_metrics(result_data, args, model_results):
    cal_result_data, test_result_data = train_test_split(
        result_data, train_size=args.cal_ratio, random_state=42
    )
    # print(len(result_data), len(cal_result_data), len(test_result_data))

    test_id_to_answer = {}
    for row in test_result_data:
        test_id_to_answer[str(row["id"])] = row["answer"]

    acc, E_ratio, F_ratio = cal_acc(test_result_data)
    model_results['acc'].append(acc)
    model_results['E_ratio'].append(E_ratio)
    model_results['F_ratio'].append(F_ratio)
    # print(acc, E_ratio, F_ratio)

    pred_sets_LAC = LAC_CP(cal_result_data, test_result_data, alpha=args.alpha)
    coverage_all_LAC = cal_coverage(pred_sets_LAC, test_id_to_answer)
    model_results["coverage_all_LAC"].append(coverage_all_LAC)
    # print('coverage_all_LAC:', coverage_all_LAC)
    set_sizes_LAC = cal_set_size(pred_sets_LAC)
    model_results["set_sizes_LAC"].append(set_sizes_LAC)
    # print('set_sizes_LAC:', set_sizes_LAC)
    uacc_LAC = acc * np.sqrt(len(ALL_OPTIONS)) / set_sizes_LAC
    model_results["uacc_LAC"].append(uacc_LAC)
    # print('uacc_LAC:', uacc_LAC)

    pred_sets_APS = APS_CP(cal_result_data, test_result_data, alpha=args.alpha)
    coverage_all_APS = cal_coverage(pred_sets_APS, test_id_to_answer)
    model_results["coverage_all_APS"].append(coverage_all_APS)
    # print('coverage_all_APS:', coverage_all_APS)
    set_sizes_APS = cal_set_size(pred_sets_APS)
    model_results["set_sizes_APS"].append(set_sizes_APS)
    # print('set_sizes_APS:', set_sizes_APS)
    uacc_APS = acc * np.sqrt(len(ALL_OPTIONS)) / set_sizes_APS
    model_results["uacc_APS"].append(uacc_APS)

    ece = get_ce(result_data=result_data, norm='l1')
    mce = get_ce(result_data=result_data, norm='max')

    model_results["ece"].append(ece)
    model_results["mce"].append(mce)
    # print('uacc_APS:', uacc_APS)

    model_results["set_sizes"].append(np.mean([set_sizes_LAC, set_sizes_APS]))
    model_results["coverage"].append(np.mean([coverage_all_LAC, coverage_all_APS]))
    model_results["uacc"].append(np.mean([uacc_LAC, uacc_APS]))


    # Call the new Abstention_CP function
    pred_outputs = Abstention_CP(cal_result_data, test_result_data, args)

    correct_predictions = 0
    total_predictions = 0
    abstentions = 0
    set_sizes = []

    for idx, prediction in pred_outputs.items():
        true_answer = test_id_to_answer[idx]
        if prediction == 'abstain':
            abstentions += 1
            continue
        elif isinstance(prediction, list):
            set_sizes.append(len(prediction))
            if true_answer in prediction:
                correct_predictions += 1
            total_predictions += 1
        else:
            if prediction == true_answer:
                correct_predictions += 1
            total_predictions += 1

    # Compute metrics
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
    else:
        accuracy = None  # No predictions made
    abstention_rate = abstentions / len(test_result_data)
    average_set_size = np.mean(set_sizes) if set_sizes else 0

    # Store metrics
    model_results['abstention_rate'].append(abstention_rate)
    model_results['accuracy'].append(accuracy)
    model_results['average_set_size'].append(average_set_size)
    
    return model_results

def calculate_metrics_for_model(model_name, args):
    model_results = defaultdict(list)
    for dataset_name in DATASETS:
        file_name = f'{args.result_data_path}/{model_name}/{dataset_name}.pkl'
        with open(file_name, 'rb') as f:
            result_data = pickle.load(f)

        model_results = calculate_metrics(result_data, args, model_results)
    return model_results


def calculate_metrics_for_seedbench(model_name, args):
    seedbench_results = defaultdict(list)
    file_name = f'{args.result_data_path}/{model_name}/seedbench.pkl'
    with open(file_name, 'rb') as f:
        result_data = pickle.load(f)
        for i in range(1, 10):
            cat_data = [row for row in result_data if row['question_type_id'] == i]
            print(f"Category {i}, length: ", len(cat_data))
            seedbench_results = calculate_metrics(cat_data, args, seedbench_results)
    return seedbench_results


def calculate_metrics_for_oodcv(model_name, args):
    oodcv_results = defaultdict(list)
    file_name = f'{args.result_data_path}/{model_name}/oodcv.pkl'
    with open(file_name, 'rb') as f:
        result_data = pickle.load(f)
        for situation in OODCV_CATS:
            cat_data = [row for row in result_data if row['situation'] == situation]
            oodcv_results = calculate_metrics(cat_data, args, oodcv_results)
    return oodcv_results


def main(args):

    full_result = defaultdict(dict)
    model_names = os.listdir(args.result_data_path)
    for model_name in tqdm(model_names):
        if args.mode == "all":
            model_metrics = calculate_metrics_for_model(model_name, args)
            full_result[model_name] = model_metrics
        elif args.mode == "seedbench":
            model_metrics = calculate_metrics_for_seedbench(model_name, args)
            full_result[model_name] = model_metrics
        elif args.mode == "oodcv":
            model_metrics = calculate_metrics_for_oodcv(model_name, args)
            full_result[model_name] = model_metrics
        else:
            raise ValueError(f"Unrecognized mode: {args.mode}")
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
    parser.add_argument("--beta", type=float, default=0.05,
                        help="The acceptable abstention rate.")
    parser.add_argument("--cum_prob_threshold", type=float, default=0.9,
                        help="Cumulative probability threshold for prediction sets.")
    parser.add_argument("--lambda1", type=float, default=0.5,
                        help="Weight for average set size in the cost function.")
    parser.add_argument("--lambda2", type=float, default=0.5,
                        help="Weight for abstention rate in the cost function.")
    args = parser.parse_args()

    main(args)
    #python -m uncertainty_quantification_via_cp --result_data_path 'output' --file_to_write 'full_result.json'
    #python -m uncertainty_quantification_via_cp --result_data_path 'output' --mode 'seedbench' --file_to_write 'full_result_seedbench.json'
    #python -m uncertainty_quantification_via_cp --result_data_path 'output' --mode 'oodcv' --file_to_write 'full_result_oodcv.json'