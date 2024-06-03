# time_series_framework/app.py
import copy
import matplotlib.pyplot as plt

save_dir = "D:/Master/SS_2024_Thesis_ISA/Thesis/Work-docs/RAMS-TSAD/Mononito/trained_models/anomaly_archive/031_UCR_Anomaly_DISTORTEDInternalBleeding20/"
# algorithm_list = ['DGHL', 'LSTMVAE', 'GMM', 'KDE']
# algorithm_list_instances = ['DGHL_1', 'DGHL_2', 'DGHL_3', 'DGHL_4', 'LSTMVAE_1', 'LSTMVAE_2', 'LSTMVAE_3', 'LSTMVAE_4', 'GMM_1', 'GMM_2', 'GMM_3', 'GMM_4', 'KDE_1', 'KDE_2', 'KDE_3', 'KDE_4']
algorithm_list = ['LOF', 'NN', 'RNN']
algorithm_list_instances = ['LOF_1', 'LOF_2', 'LOF_3', 'LOF_4', 'NN_1', 'NN_2', 'NN_3', 'RNN_1', 'RNN_2', 'RNN_3',
                            'RNN_4']
import os
import logging
import torch as t
from Datasets.load import load_data
from Utils.utils import get_args_from_cmdline
from Model_Training.train import TrainModels
from loguru import logger
import traceback
import numpy as np
from Model_Selection.inject_anomalies import Inject
from Metrics.Ensemble_GA import genetic_algorithm
from Model_Selection.Thompson_Sampling import (fit_thompson_sampling, initialize_windows, sample_model,
                                               update_posteriors, calculate_reward, rank_models, plot_history)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trained_models(algorithm_list, save_dir):
    """Load trained models from the save directory."""
    trained_models = {}
    for model_name in algorithm_list:
        model_path = os.path.join(save_dir, f"{model_name}.pth")
        print(model_path)

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = t.load(f)
                model.eval()  # Set model to evaluation mode
                trained_models[model_name] = model
        else:
            raise FileNotFoundError(f"Model {model_name} not found in {save_dir}")
    return trained_models


def run_app(algorithm_list, algorithm_list_instances):
    args = get_args_from_cmdline()
    data_dir = args['dataset_path']
    train_data = load_data(dataset='anomaly_archive', group='train',
                           entities='031_UCR_Anomaly_DISTORTEDInternalBleeding20', downsampling=10,
                           min_length=256, root_dir=data_dir, normalize=True, verbose=False)
    test_data = load_data(dataset='anomaly_archive', group='test',
                          entities='031_UCR_Anomaly_DISTORTEDInternalBleeding20', downsampling=10,
                          min_length=256, root_dir=data_dir, normalize=True, verbose=False)

    # Ensure data is correctly loaded
    if not train_data.entities:
        logger.error("Failed to load training data. Please check the dataset and paths.")
        return

    if not test_data.entities:
        logger.error("Failed to load test data. Please check the dataset and paths.")
        return
    entity = '031_UCR_Anomaly_DISTORTEDInternalBleeding20'
    dataset = 'anomaly_archive'
    model_trainer = TrainModels(dataset=dataset,
                                entity=entity,
                                algorithm_list=algorithm_list,
                                downsampling=args['downsampling'],
                                min_length=args['min_length'],
                                root_dir=args['dataset_path'],
                                training_size=args['training_size'],
                                overwrite=args['overwrite'],
                                verbose=args['verbose'],
                                save_dir=args['trained_model_path'])
    try:
        model_trainer.train_models(model_architectures=args['model_architectures'])

        # Load trained models

        global trained_models
        trained_models = load_trained_models(algorithm_list_instances, save_dir)

        if not trained_models:
            raise ValueError("No models were loaded. Please check the model paths and ensure models are trained.")

        anomaly_list = ['spikes']

        # Use Thompson Sampling with Epsilon Greedy for model selection
        print(f'test_data_before.entities[0].Y: \n {np.size(test_data.entities[0].Y)}')
        print(f'test_data_before.entities[0].labels: \n {test_data.entities[0].labels}')
        print(f'test_data_before.entities[0].X: \n {test_data.entities[0].X}')
        print(f'test_data_before.entities[0].n_time : \n {test_data.entities[0].n_time}')
        print(f'test_data_before.entities[0].mask : \n {np.size(test_data.entities[0].mask)}')
        print(f'test_data_before.entities[0].verbose : \n {test_data.entities[0].verbose}')
        print(f'test_data_before.entities[0].n_exogenous : \n {test_data.entities[0].n_exogenous}')
        print(f'test_data_before.entities[0].n_features : \n {test_data.entities[0].n_features}')
        test_data_before = copy.deepcopy(test_data)
        train_data_before = copy.deepcopy(train_data)
        train_data, anomaliy_sizes_train = Inject(train_data, anomaly_list)
        test_data, anomaly_sizes = Inject(test_data, anomaly_list)

        print(f'test_data.entities[0].Y: \n {np.size(test_data.entities[0].Y)}')
        print(f'test_data.entities[0].labels: \n {test_data.entities[0].labels}')
        print(f'test_data.entities[0].X: \n {test_data.entities[0].X}')
        print(f'test_data.entities[0].n_time : \n {test_data.entities[0].n_time}')
        print(f'test_data.entities[0].mask : \n {np.size(test_data.entities[0].mask)}')
        print(f'test_data.entities[0].verbose : \n {test_data.entities[0].verbose}')
        print(f'test_data.entities[0].n_exogenous : \n {test_data.entities[0].n_exogenous}')
        print(f'test_data.entities[0].n_features : \n {test_data.entities[0].n_features}')
        anomaly_start = np.argmax(test_data.entities[0].labels)
        anomaly_end = test_data.entities[0].Y.shape[1] - np.argmax(test_data.entities[0].labels[::-1])
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20, 6))
        axes[0].plot(test_data.entities[0].Y.flatten(), color='darkblue')
        axes[0].plot(np.arange(anomaly_start, anomaly_end),
                     test_data.entities[0].Y.flatten()[anomaly_start:anomaly_end],
                     color='red')
        axes[0].plot(np.arange(anomaly_start, anomaly_end),
                     test_data_before.entities[0].Y.flatten()[anomaly_start:anomaly_end], color='darkblue',
                     linestyle='--')
        axes[0].set_title('Test data with Injected Anomalies', fontsize=16)
        axes[1].plot(anomaly_sizes.flatten(), color='pink')
        axes[1].plot(test_data.entities[0].labels.flatten(), color='red')
        axes[1].set_title('Anomaly Scores', fontsize=16)
        plt.show()

        # priors, history = fit_thompson_sampling(
        #     dataset=test_data,
        #     models=trained_models,
        #     data=test_data.entities[0].Y,
        #     targets=test_data.entities[0].labels,
        #     initial_epsilon=0.1,
        #     epsilon_decay=0.99,
        #     f1_weight=0.5,
        #     pr_auc_weight=0.5,
        #     iterations=10
        # )
        # Rank models based on the final priors
        # ranked_models = rank_models(priors)
        # print("Ranked Models:", ranked_models)

        # Plot the history of model rankings
        # plot_history(history, trained_models)
        # Run genetic algorithm for model selection
        best_ensemble, best_f1, best_pr_auc, best_fitness = genetic_algorithm(dataset, entity, train_data, test_data,
                                                                              algorithm_list_instances, trained_models,
                                                                              population_size=5, generations=5,
                                                                              meta_model_type='lr', mutation_rate=0.2)
        logger.info(
            f"Best ensemble: {best_ensemble} with F1 score {best_f1}, PR AUC {best_pr_auc}, and fitness {best_fitness}")
        print(
            f"Best ensemble: {best_ensemble} with F1 score {best_f1}, PR AUC {best_pr_auc}, and fitness {best_fitness}")
        # print(test_data.entities[0].Y)

    except Exception as e:
        logger.info(f'Traceback for Entity: {entity} Dataset: {dataset}')
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    run_app(algorithm_list, algorithm_list_instances)
