import torch

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
        # evaluator
        from core.evaluator import Evaluator
        self.evaluator = Evaluator(device='cpu')
        # trainer
        from core.trainer import Trainer
        self.trainer = Trainer
        # plotter
        from utils.plotter import Plotter
        save_plots = config['plotting']['save_plots']
        save_dir = config['plotting']['save_dir']
        self.plotter = Plotter(save_plots, save_dir)
        # logger
        from utils.logger import Logger
        self.logger = Logger(config)
        self.logger.log("Initialized GaussianClassificationExperiment.")

    def run(self):
        # create an MLP model
        from models.mlp import MLP
        model = MLP(input_dim=self.model_config['input_dim'],
                    hidden_layers=self.model_config['hidden_layers'],
                    output_dim=self.model_config['output_dim'],
                    activation=self.model_config['activation'],
                    bias=self.model_config['bias'])
        self.logger.log(f"Created MLP model: {model}")

        # create test dataset and test dataloader
        from core.dataset import Gaussian
        test_dataset = Gaussian(feature_dim=self.dataset_config['feature_dim'],
                                n_samples_per_class=self.dataset_config['test_size']//2,
                                mean_distance=self.dataset_config['mean_distance'],
                                sigma=self.dataset_config['sigma'],
                                seed=self.exp_config['seed'])
        self.logger.log(f"Created test dataset with {len(test_dataset)} samples.")
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        self.logger.log("Created test DataLoader.")

        # Estimate classifier density D(E)
        true_errors = self.estimate_classifier_density(model, test_loader)
        self.logger.log(f"Estimated classifier density D(E) with {len(true_errors)} samples.")
        hist_fig, _ = self.plotter.plot_histogram(data=true_errors,
                                                  bins=self.doc_config['histogram_bins'],
                                                  title = "Classifier Density D(E)",
                                                  xlabel = "True Error",
                                                  ylabel = "Density")
        self.logger.save_figure(hist_fig, "classifier_density_histogram.png")
        
        """
        # test error distribution for random weights with zero training error
        zero_empirical_true_errors = []  
        for n_train_samples in self.erm_config['n_values']:
            train_dataset = Gaussian(feature_dim=self.dataset_config['feature_dim'],
                                     n_samples_per_class=n_train_samples//2,
                                     mean_distance=self.dataset_config['mean_distance'],
                                     sigma=self.dataset_config['sigma'],
                                     seed=self.exp_config['seed'])
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            self.trainer.random_sample_to_zero_error()
            true_error = self.evaluator.compute_error(model, test_loader)
            zero_empirical_true_errors.append((n_train_samples, true_error))
        # plot boxplot of true errors for different training set sizes
        self.plotter.plot_boxplot(data=zero_empirical_true_errors,
                                  title="True Error Distribution for Random Weights with Zero Training Error",
                                  xlabel="Number of Training Samples",
                                  ylabel="True Error")"""

        # Show plots   
        # self.plotter.show_plots()

    def estimate_classifier_density(self, model, data_loader) -> list[float]:
        # Estimate classifier density D(E) by sampling random weights
        n_trials = self.doc_config['n_trials']
        true_errors = []
        for _ in range(n_trials):
            flat_weights = model.sample_unit_sphere_weights()
            model.set_flatten_weights(flat_weights)
            true_error = self.evaluator.compute_error(model, data_loader)
            true_errors.append(true_error)
        return true_errors
        
    
    
        