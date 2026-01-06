import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

import numpy as np
from datetime import datetime as dt
from tqdm import tqdm

# DataLoader
from torch.utils.data import DataLoader

# datasets
from core.dataset import Gaussian

# models 
from models.mlp import MLP

class BaseExperiment:
    def __init__(self):
        ...
    def run(self):
        ...
        
class ExperimentFactory:
    @staticmethod
    def create_experiment(config) -> BaseExperiment:
        # create and return the corresponding experiment instance based on config
        experiment_type = config['experiment']['type']
        if experiment_type == 'gaussian_classification':
            from core.experiments import GaussianClassificationExperiment
            return GaussianClassificationExperiment(config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        

class GaussianClassificationExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__()
        # experiment configuration
        self.exp_config = config['experiment']
        # dataset configuration
        self.dataset_config = config['dataset']['gaussian']
        # model configuration
        if self.exp_config['model'] in config['models']:
            self.model_config = config['models'][self.exp_config['model']]
        else:
            raise ValueError(f"Model {self.exp_config['model']} not found in config models.")
        # DOC configuration
        self.doc_config = config['doc']
        # ERM configurations
        self.erm_config = config['erm']

        # logger
        from utils.logger import Logger
        self.logger = Logger(config)
        self.logger.log("Initialized GaussianClassificationExperiment.")

        # evaluator
        from core.evaluator import Evaluator
        
        self.evaluator = Evaluator(device='cuda')
        self.logger.log(f"Using {torch.cuda.get_device_name(0)} for evaluation.")
            
        # trainer
        from core.trainer import Trainer
        self.trainer = Trainer()
        # plotter
        from utils.plotter import Plotter
        save_plots = config['plotting']['save_plots']
        save_dir = config['plotting']['save_dir']
        self.plotter = Plotter(save_plots, save_dir)
        

    def run(self):
        start_time = dt.now()
        # create an MLP model
        model = MLP(input_dim=self.model_config['input_dim'],
                    hidden_layers=self.model_config['hidden_layers'],
                    output_dim=self.model_config['output_dim'],
                    activation=self.model_config['activation'],
                    bias=self.model_config['bias'])
        self.logger.log(f"Created MLP model: {model}")

        # move model once to evaluation device (avoid doing it inside every compute_error call)
        model.to(self.evaluator.device)

        # create test dataset and test dataloader
        test_dataset = Gaussian(feature_dim=self.dataset_config['feature_dim'],
                                n_samples_per_class=self.dataset_config['test_size']//2,
                                mean_distance=self.dataset_config['mean_distance'],
                                sigma=self.dataset_config['sigma'],
                                seed=self.exp_config['seed'])
        self.logger.log(f"Created test dataset with {len(test_dataset)} samples.")
        test_loader = DataLoader(test_dataset, batch_size=512, num_workers=4, pin_memory=True, shuffle=False)
        self.logger.log("Created test DataLoader.")
        
        # Estimate classifier density D(E)
        self.logger.log(f"Estimating classifier density D(E) with {self.doc_config['n_trials']} trials.")
        true_errors = self.estimate_classifier_density(model, test_loader)
        self.logger.save_numpy_array(np.array(true_errors), "classifier_density.npy")
        self.logger.log(f"Estimating classifier density completed.")
        hist_fig, _ = self.plotter.plot_histogram(data=true_errors,
                                                  bins=self.doc_config['histogram_bins'],
                                                  title = "Classifier Density D(E)",
                                                  xlabel = "True Error",
                                                  ylabel = "Density")
        self.logger.save_figure(hist_fig, "classifier_density_histogram.png")
        
        """
        # test error distribution for random weights with zero training error
        self.logger.log("Estimating true error distribution for random weights with zero training error.")
        solutions_true_errors = self.estimate_true_error_distribution(model, test_loader)
        # Save numpy array of zero empirical true errors
        self.logger.save_numpy_array(np.array(solutions_true_errors, dtype=object), "solutions_true_errors.npy")
        # plot boxplot of true errors for different training set sizes
        boxplot_fig, _ = self.plotter.plot_boxplot(data=solutions_true_errors,
                                                    title="True Error Distribution for Random Weights with Zero Training Error",
                                                    xlabel="Number of Training Samples",
                                                    ylabel="True Error")
        self.logger.save_figure(boxplot_fig, "solutions_true_error_boxplot.png")
        # Show plots   
        # self.plotter.show_plots()
        """
        end_time = dt.now()
        self.logger.log(f"Experiment completed in {(end_time - start_time)}.")

    def estimate_classifier_density(self, model, data_loader) -> list[float]:
        # Estimate classifier density D(E) by sampling random weights
        n_trials = self.doc_config['n_trials']
        true_errors = []
        # model should already be on evaluation device
        for _ in tqdm(range(n_trials)):
            flat_weights = model.sample_unit_sphere_weights(device=self.evaluator.device)
            model.set_flatten_weights(flat_weights)
            true_error = self.evaluator.compute_error(model, data_loader)
            true_errors.append(true_error)
        return true_errors
    
    def estimate_true_error_distribution(self, model, test_loader) -> list[float]:
        # Estimate true error distribution for random weights with zero training error
        n_values = self.erm_config['n_values']
        solutions_per_n = self.erm_config['solutions_per_n']
        solutions_true_errors = []  # list of tuples (n_train_samples, true_error)
        # ensure model is on the evaluator device
        model.to(self.evaluator.device)
        for n_train_samples in n_values:
            self.logger.log(f"Finding zero empirical error solutions for {n_train_samples} training samples.")
            for s in tqdm(range(solutions_per_n)):
                if n_train_samples==0:
                    # if zero training samples, just sample random weights and compute true error
                    flat_weights = model.sample_unit_sphere_weights(device=self.evaluator.device)
                    model.set_flatten_weights(flat_weights)
                    true_error = self.evaluator.compute_error(model, test_loader)
                    solutions_true_errors.append((n_train_samples, true_error))
                    continue

                # create train dataset and train dataloader
                train_dataset = Gaussian(feature_dim=self.dataset_config['feature_dim'],
                                        n_samples_per_class=n_train_samples//2,
                                        mean_distance=self.dataset_config['mean_distance'],
                                        sigma=self.dataset_config['sigma'],
                                        seed=self.exp_config['seed'])
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

                self.trainer.sample_unit_sphere_weights_until_zero_error(model, train_loader, self.evaluator)
                true_error = self.evaluator.compute_error(model, test_loader)
                solutions_true_errors.append((n_train_samples, true_error))
        return solutions_true_errors
