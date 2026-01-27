import argparse, yaml
from experiments.base_experiment import BaseExperiment

class ExperimentFactory:
    @staticmethod
    def create_experiment(config) -> BaseExperiment:
        # create and return the corresponding experiment instance based on config
        experiment_type = config['experiment']['type']
        if experiment_type == 'gaussian_classification':
            from experiments.gaussian_classification_experiment import GaussianClassificationExperiment
            return GaussianClassificationExperiment(config)
        elif experiment_type == 'mnist_classification':
            from experiments.mnist_classification_experiment import MnistClassificationExperiment
            return MnistClassificationExperiment(config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment = ExperimentFactory.create_experiment(config)
    experiment.run()


if __name__ == "__main__": 
    main()