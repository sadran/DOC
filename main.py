import argparse, yaml
from core.experiments import ExperimentFactory



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