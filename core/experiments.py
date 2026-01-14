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
        # -----------------------------
        # 1) Build model (A1/A2/A3 etc.)
        # -----------------------------
        model = MLP(input_dim=self.model_config['input_dim'],
                    hidden_layers=self.model_config['hidden_layers'],
                    output_dim=self.model_config['output_dim'],
                    activation=self.model_config['activation'],
                    bias=self.model_config['bias'])
        self.logger.log(f"Created MLP model: {model}")

        # move model once to evaluation device (avoid doing it inside every compute_error call)
        model.to(self.evaluator.device)

        # -----------------------------------------
        # 2) Build a fixed balanced test set + loader
        # -----------------------------------------
        test_dataset = Gaussian(feature_dim=self.dataset_config['feature_dim'],
                                n_samples_per_class=self.dataset_config['test_size']//2,
                                mean_distance=self.dataset_config['mean_distance'],
                                sigma=self.dataset_config['sigma'],
                                seed=self.exp_config['seed'])
        self.logger.log(f"Created test dataset with {len(test_dataset)} samples.")
        test_loader = DataLoader(test_dataset, batch_size=512, num_workers=4)
        self.logger.log("Created test DataLoader.")
        
        # ---------------------------------------------
        # 3) Estimate classifier density D(E) (left plot)
        # ---------------------------------------------
        self.logger.log(f"Estimating classifier density D(E) with {self.doc_config['n_trials']} trials.")
        true_errors = self.estimate_classifier_density(model, test_loader)
        self.logger.save_numpy_array(np.array(true_errors), "classifier_density.npy")
        self.logger.log(f"Estimating classifier density completed.")
        hist_fig, _ = self.plotter.plot_histogram(data=true_errors,
                                                  bins=self.doc_config['histogram_bins'],
                                                  title = "Classifier Density D(E)",
                                                  xlabel = "E",
                                                  ylabel = "D(E)")
        self.logger.save_figure(hist_fig, "classifier_density_histogram.png")
        
        # -------------------------------------------------------------------
        # 4) Estimate true-error distribution of ERM solutions (middle plot)
        # -------------------------------------------------------------------
        self.logger.log("Estimating true error distribution for random weights with zero training error.")
        solutions_true_errors = self.estimate_true_error_distribution(model, test_loader)
        # Save numpy array of zero empirical true errors
        self.logger.save_numpy_array(np.array(solutions_true_errors, dtype=object), "solutions_true_errors.npy")
        # plot boxplot of true errors for different training set sizes
        boxplot_fig, _ = self.plotter.plot_boxplot(true_errors=solutions_true_errors,
                                                    n_values=self.erm_config['n_values'],
                                                    title="True Error Distribution for Random Weights with Zero Training Error",
                                                    xlabel="Number of Training Samples",
                                                    ylabel="True Error")
        self.logger.save_figure(boxplot_fig, "solutions_true_error_boxplot.png")
        
         # -------------------------------------------------------------------
        # 5) Right-column plot (red x vs blue x)
        #     - red x: empirical mean of ERM true errors (from middle plot)
        #     - blue x: DOC-based predicted mean computed from D(E) (left plot)
        # -------------------------------------------------------------------
        # Red crosses: empirical mean test error for each n
        erm_means = np.array([float(np.mean(errs)) for errs in solutions_true_errors], dtype=float)
        # Blue crosses: DOC prediction from D(E)
        doc_means = self.doc_predicted_mean_error(true_errors)
        # Plot comparison (right-column figure)
        doc_vs_erm_fig, ax = self.plotter.plot_doc_vs_erm(n_values, erm_means, doc_means)
        self.logger.save_figure(doc_vs_erm_fig, "doc_vs_erm_mean_true_error.png")

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
        true_errors = []  # list[list[float]]
        # ensure model is on the evaluator device
        model.to(self.evaluator.device)
        for n in n_values:
            errors_for_n = []
            self.logger.log(f"Finding zero empirical error solutions for {n} training samples.")
            for s in tqdm(range(solutions_per_n)):
                if n==0:
                    # if zero training samples, just sample random weights and compute true error
                    flat_weights = model.sample_unit_sphere_weights(device=self.evaluator.device)
                    model.set_flatten_weights(flat_weights)
                    true_error = self.evaluator.compute_error(model, test_loader)
                    errors_for_n.append(true_error)
                    continue

                # create train dataset and train dataloader
                train_dataset = Gaussian(feature_dim=self.dataset_config['feature_dim'],
                                        n_samples_per_class=n//2,
                                        mean_distance=self.dataset_config['mean_distance'],
                                        sigma=self.dataset_config['sigma'],
                                        seed=self.exp_config['seed'])
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

                self.trainer.sample_unit_sphere_weights_until_zero_error(model, train_loader, self.evaluator)
                true_error = self.evaluator.compute_error(model, test_loader)
                errors_for_n.append(true_error)
            true_errors.append(errors_for_n)
        return true_errors
    
    def doc_predicted_mean_error(self, true_errors: list[float], n_values: list[int], bins: int = 100):
        """
        Compute the DOC-based predicted mean true error for each n using the sampled true_errors.

        Returns:
            pred: np.ndarray of shape (len(n_values),)
        """
        n_values = self.erm_config["n_values"]
        bins = self.doc_config["histogram_bins"]
        # Compute histogram-based density and discrete approximation of the DOC formula.
        hist, bin_edges = np.histogram(
            true_errors,
            bins= bins,
            range=(0.0, 1.0),
            density=True,
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        dE = float(bin_edges[1] - bin_edges[0])

        doc_means = []
        for n in n_values:
            weights = (1.0 - bin_centers) ** int(n)
            numerator = np.sum(bin_centers * weights * hist) * dE
            denominator = np.sum(weights * hist) * dE
            doc_means.append(numerator / denominator)
        doc_means = np.array(doc_means, dtype=float)
        return doc_means
