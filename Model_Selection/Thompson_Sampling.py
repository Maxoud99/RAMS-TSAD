import random
import numpy as np
from scipy.stats import beta
from loguru import logger
from Algorithms.base_model import PyMADModel
from sklearn.metrics import f1_score, precision_recall_curve, auc
from typing import Tuple, Union, List
from Loaders.loader import Loader
from Datasets.dataset import Dataset, Entity
from Metrics.Ensemble_GA import evaluate_model_consistently
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from Utils.model_selection_utils import evaluate_model
from Metrics.metrics import range_based_precision_recall_f1_auc
import traceback
from Utils.utils import de_unfold
import torch as t

from typing import List, Tuple
import numpy as np


def initialize_windows(data: np.ndarray, targets: np.ndarray, mask: np.ndarray, n_windows: int) -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    if data.size == 0 or targets.size == 0:
        raise ValueError("Data and targets must not be empty.")

    if n_windows <= 0:
        raise ValueError("Number of windows must be greater than zero.")

    total_size = data.shape[1]  # We use shape[1] since the data has shape (1, n)
    window_size = total_size // n_windows
    remainder = total_size % n_windows

    data_windows = []
    targets_windows = []
    masks_windows = []

    start_index = 0

    for i in range(n_windows):
        end_index = start_index + window_size + (1 if i < remainder else 0)
        data_windows.append(data[:, start_index:end_index])
        targets_windows.append(targets[:, start_index:end_index])
        masks_windows.append(mask[:, start_index:end_index])
        start_index = end_index

    return data_windows, targets_windows, masks_windows


def sample_model(models: Dict[str, Any], priors: Dict[str, List[float]], epsilon: float) -> str:
    if random.random() < epsilon:
        chosen_model = random.choice(list(models.keys()))
        logger.info(f"Epsilon-Greedy: Randomly chosen model {chosen_model}")
    else:
        samples = {model_name: beta.rvs(a=prior[0], b=prior[1])
                   for model_name, prior in priors.items()}
        chosen_model = max(samples, key=samples.get)
        logger.info(f"Thompson Sampling: Chosen model {chosen_model} with sample value {samples[chosen_model]}")
    return chosen_model


def update_posteriors(priors: Dict[str, List[float]], model_name: str, reward: float) -> None:
    if model_name not in priors:
        raise ValueError(f"Model name {model_name} not found in priors.")

    priors[model_name][0] += reward
    priors[model_name][1] += (1 - reward)
    logger.info(f"Updated priors for model {model_name}: {priors[model_name]}")


def calculate_reward(f1: float, pr_auc: float, f1_weight: float, pr_auc_weight: float) -> float:
    return (f1_weight * f1) + (pr_auc_weight * pr_auc)


def fit_thompson_sampling(dataset,
                          models: Dict[str, Any], data: np.ndarray, targets: np.ndarray, initial_epsilon: float = 0.1,
                          epsilon_decay: float = 0.99, f1_weight: float = 0.5, pr_auc_weight: float = 0.5,
                          iterations: int = 100) -> Tuple[Dict[str, List[float]], List[Dict[str, float]]]:
    mask = dataset.entities[0].mask
    print(f"Data shape before windowing: {data.shape}")
    print(f"Targets shape before windowing: {targets.shape}")
    print(f"Mask shape before windowing: {mask.shape}")

    n_times = dataset.entities[0].n_time
    dataset.entities[0].n_time = n_times // iterations
    dataset.total_time = n_times // iterations
    data_windows, targets_windows, New_mask = initialize_windows(data, targets, mask, iterations)

    priors = {model_name: [1, 1] for model_name in models}
    epsilon = initial_epsilon
    history = []

    for iteration in range(iterations):
        logger.info(f"Iteration {iteration + 1}")
        chosen_model_name = sample_model(models, priors, epsilon)
        chosen_model = models[chosen_model_name]

        X_test_window = data_windows[iteration]
        y_test_window = targets_windows[iteration]
        masks_window = New_mask[iteration]

        print(f"First window data shape: {X_test_window[0].shape}")
        print(f"First window target shape: {y_test_window[0].shape}")
        print(f"First window mask shape: {masks_window[0].shape}")

        dataset.entities[0].Y = X_test_window
        dataset.entities[0].labels = targets_windows[iteration]
        dataset.entities[0].mask = masks_window

        print(f'test_data.entities[0].Y: \n {dataset.entities[0].Y}')
        print(f'test_data.entities[0].labels: \n {dataset.entities[0].labels}')
        print(f'test_data.entities[0].X: \n {dataset.entities[0].X}')
        print(f'test_data.entities[0].n_time : \n {dataset.entities[0].n_time}')
        print(f'test_data.entities[0].mask : \n {dataset.entities[0].mask}')
        print(f'test_data.entities[0].verbose : \n {dataset.entities[0].verbose}')
        print(f'test_data.entities[0].n_exogenous : \n {dataset.entities[0].n_exogenous}')
        print(f'test_data.entities[0].n_features : \n {dataset.entities[0].n_features}')

        try:
            y_true, y_scores, y_true_dict, y_scores_dict = evaluate_model_consistently(dataset, chosen_model, chosen_model_name)

            _, _, f1, pr_auc, *_ = range_based_precision_recall_f1_auc(y_true, y_scores)
            reward = calculate_reward(f1, pr_auc, f1_weight, pr_auc_weight)
            update_posteriors(priors, chosen_model_name, reward)
            logger.info(
                f"Window {iteration + 1}: Model {chosen_model_name} - F1 Score = {f1}, PR AUC = {pr_auc}, Reward = {reward}")
            logger.info(f"Priors: {priors}")

        except Exception as e:
            logger.error(f"Error evaluating model {chosen_model_name}: {e}")
            detailed_traceback = traceback.format_exc()
            print(detailed_traceback)
            continue  # Skip the current iteration on error

        epsilon *= epsilon_decay

        history.append({model_name: prior[0] / (prior[0] + prior[1]) for model_name, prior in priors.items()})
        logger.info(f"Finished iteration {iteration + 1}")

    return priors, history


def rank_models(priors: Dict[str, List[float]]) -> List[Tuple[str, float]]:
    """
    Rank the models based on their Beta distribution means.

    Parameters:
    priors (Dict[str, List[float]]): Dictionary of Beta distribution priors for each model.

    Mean Calculation: For each model, the mean of the Beta distribution is calculated as α/(α+β).

    Returns:
    List[Tuple[str, float]]: List of models and their beta scores, sorted from highest to lowest.
    """
    model_ranking = {model_name: prior[0] / (prior[0] + prior[1]) for model_name, prior in priors.items()}
    ranked_models = sorted(model_ranking.items(), key=lambda x: x[1], reverse=True)
    return ranked_models


def plot_history(history: List[Dict[str, float]], models: Dict[str, Any]) -> None:
    plt.figure(figsize=(12, 8))
    for model_name in models.keys():
        scores = [h[model_name] for h in history]
        plt.plot(scores, label=model_name)
    plt.xlabel('Iteration')
    plt.ylabel('Beta Mean')
    plt.title('Evolution of Model Rankings')
    plt.legend()
    plt.show()
