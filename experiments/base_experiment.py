import torch
from core.evaluator import Evaluator
from utils.logger import Logger
from utils.plotter import Plotter
from core.trainer import Trainer

class BaseExperiment:
    def __init__(self, config):
        # Store the config for use in subclasses
        self.config = config

        #-----------------------------
        # 2) Initialize components
        #-----------------------------
        self.logger = Logger(config)
        self.logger.log(f"Initializing {self.__class__.__name__}.")
        
        # evaluator
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.evaluator = Evaluator(device=device)
        if device == 'cuda':
            self.logger.log(f"Using {torch.cuda.get_device_name(0)} for evaluation.")
        else:
            self.logger.log("Using CPU for evaluation.")
            
        # trainer
        self.trainer = Trainer()

        # plotter
        self.plotter = Plotter()

    def run(self):
        ...
        
