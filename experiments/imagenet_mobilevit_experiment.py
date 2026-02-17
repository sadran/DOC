import torch

import numpy as np
from datetime import datetime as dt
from tqdm import tqdm

# base experiment
from experiments.base_experiment import BaseExperiment
# DataLoader
from torch.utils.data import DataLoader
# datasets
from core.datasets import ImageNet1k
# models 
from core.models import MobileViT



class ImageNetMobileViTExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)
        # -----------------------------------------
        # 1) Create the model
        # -----------------------------------------
         # model configuration
        if self.config['experiment']['model'] in config['models']:
            model_config = config['models'][self.config['experiment']['model']]
        else:
            raise ValueError(f"Model {self.config['experiment']['model']} not found in config models.")
        # create model
        self.model = MobileViT(model_name=model_config['name'], num_classes=model_config['num_classes'])
        self.logger.log(f"Created the model: {self.model}")
        self.model.to(self.evaluator.device)

        # -----------------------------------------
        # 2) Build a fixed balanced test set + loader
        # -----------------------------------------
        #self.test_dataset = ImageNet1k(self.config['dataset']['root_path'], split='test', n_samples=self.config['dataset']['test_size'])
        self.test_dataset = ImageNet1k(data_root_dir="data/ILSVRC2012_img/ILSVRC2012_img_train",
                                       split='test',
                                      n_samples=self.config['dataset']['test_size'])
        self.logger.log(f"Loaded test dataset with {len(self.test_dataset)} samples from ImageNet1k dataset.")
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.config['dataset']['test_batch_size'],
                                      num_workers=4,
                                      pin_memory=True,
                                      shuffle=False)
        self.logger.log("Created test DataLoader.")

        self.logger.log(f"Initializing {self.__class__.__name__} completed.")
        

    def run(self):
        start_time = dt.now()
        # ---------------------------------------------
        # 1) Estimate classifier density D(E) (left plot)
        # ---------------------------------------------
        self.logger.log(f"Estimating classifier density D(E) with {self.config['doc']['n_trials']} trials.")
        true_errors = self.estimate_classifier_density()
        self.logger.save_numpy_array(np.array(true_errors), "classifier_density.npy")
        self.logger.log(f"Estimating classifier density completed.")
        hist_fig, _ = self.plotter.plot_histogram(data=true_errors,
                                                  bins=self.config['doc']['histogram_bins'],
                                                  title = "Classifier Density D(E)",
                                                  xlabel = "E",
                                                  ylabel = "D(E)")
        self.logger.save_figure(hist_fig, "classifier_density_histogram.png")
        
        # -------------------------------------------------------------------
        # 2) Estimate true-error distribution of ERM solutions (middle plot)
        # -------------------------------------------------------------------
        self.logger.log("Estimating true error distribution for random weights with zero training error.")
        solutions_true_errors = self.estimate_true_error_distribution()
        # Save numpy array of zero empirical true errors
        self.logger.save_numpy_array(np.array(solutions_true_errors, dtype=object), "solutions_true_errors.npy")
        # plot boxplot of true errors for different training set sizes
        boxplot_fig, _ = self.plotter.plot_boxplot(true_errors=solutions_true_errors,
                                                    n_values=self.config['erm']['n_values'],
                                                    title="True Error Distribution for Random Weights with Zero Training Error",
                                                    xlabel="Number of Training Samples",
                                                    ylabel="True Error")
        self.logger.save_figure(boxplot_fig, "solutions_true_error_boxplot.png")
        
        # -------------------------------------------------------------------
        # 3) Right-column plot (red x vs blue x)
        #     - red x: empirical mean of ERM true errors (from middle plot)
        #     - blue x: DOC-based predicted mean computed from D(E) (left plot)
        # -------------------------------------------------------------------
        self.logger.log("Computing DOC-based predicted mean true error and comparing with ERM empirical means.")
        # Red crosses: empirical mean test error for each n
        erm_means = np.array([float(np.mean(errs)) for errs in solutions_true_errors], dtype=float)
        # Blue crosses: DOC prediction from D(E)
        doc_means = self.doc_predicted_mean_error(true_errors)
        # Plot comparison (right-column figure)
        doc_vs_erm_fig, ax = self.plotter.plot_doc_vs_erm(self.config['erm']['n_values'], erm_means, doc_means)
        self.logger.save_figure(doc_vs_erm_fig, "doc_vs_erm_mean_true_error.png")

        end_time = dt.now()
        self.logger.log(f"Experiment completed in {(end_time - start_time)}.")

    def estimate_classifier_density(self) -> list[float]:
        # Estimate classifier density D(E) by sampling random weights
        n_trials = self.config['doc']['n_trials']
        true_errors = []
        # model should already be on evaluation device
        for _ in tqdm(range(n_trials)):
            flat_weights =  self.model.sample_unit_sphere_weights(device=self.evaluator.device)
            self.model.set_flatten_weights(flat_weights)
            true_error = self.evaluator.compute_error(self.model, self.test_loader)
            true_errors.append(true_error)
        return true_errors
    
    def estimate_true_error_distribution(self) -> list[float]:
        # Estimate true error distribution for random weights with zero training error
        n_values = self.config['erm']['n_values']

        solutions_per_n = self.config['erm']['solutions_per_n']
        true_errors = []  # list[list[float]]
        # ensure model is on the evaluator device
        self.model.to(self.evaluator.device)
        for n in n_values:
            errors_for_n = []
            self.logger.log(f"Finding zero empirical error solutions for {n} training samples.")
            for s in tqdm(range(solutions_per_n)):
                if n==0:
                    # if zero training samples, just sample random weights and compute true error
                    flat_weights = self.model.sample_unit_sphere_weights(device=self.evaluator.device)
                    self.model.set_flatten_weights(flat_weights)
                    true_error = self.evaluator.compute_error(self.model, self.test_loader)
                    errors_for_n.append(true_error)
                    continue

                # create train dataset and train dataloader
                train_dataset = Mnist(images_path=self.config['dataset']['test_images_filepath'],
                                 labels_path=self.config['dataset']['test_labels_filepath'],
                                 n_samples=n)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

                self.trainer.sample_unit_sphere_weights_until_zero_error(self.model, train_loader, self.evaluator)
                true_error = self.evaluator.compute_error(self.model, self.test_loader)
                errors_for_n.append(true_error)
            true_errors.append(errors_for_n)
        return true_errors
    
    def doc_predicted_mean_error(self, true_errors: list[float], bins: int = 100):
        """
        Compute the DOC-based predicted mean true error for each n using the sampled true_errors.

        Returns:
            pred: np.ndarray of shape (len(n_values),)
        """
        n_values = [n for n in range(0, 31, 2)]
        
        bins = self.config['doc']["histogram_bins"]
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
