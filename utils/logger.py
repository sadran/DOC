from pathlib import Path
from datetime import datetime as dt
import yaml
import numpy as np

class Logger:
    def __init__(self, config):
        self.config = config

        self.save_dir = Path("results", config["experiment"]["name"], dt.now().strftime("%Y%m%d_%H%M"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        with open(self.save_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

    def log(self, message: str):
        print(message)
        log_file = self.save_dir / "log.txt"
        with open(log_file, "a") as f:
            f.write(f"{dt.now().isoformat()} - {message}\n")
    
    def save_figure(self, fig, filename: str):
        fig_path = self.save_dir / filename
        fig.savefig(fig_path)

    def save_numpy_array(self, array, filename: str):
        array_path = self.save_dir / filename
        np.save(array_path, array)