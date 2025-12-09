from torch.utils.data import Dataset
import torch

class Gaussian(Dataset):
    def __init__(self, 
                 feature_dim: int = 2,
                 n_samples_per_class: int = 5000,
                 mean_distance: float = 2.0,
                 sigma: float = 0.5,
                 seed: int|None = None):
        
        super().__init__()
        
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
        else:
            g = None

        mu0 = torch.zeros(feature_dim)
        mu1 = torch.zeros(feature_dim)

        mu0[0] = -mean_distance / 2
        mu1[0] = mean_distance / 2

        x0 = torch.normal(mu0.repeat([n_samples_per_class, 1]), sigma, generator=g)
        x1 = torch.normal(mu1.repeat(n_samples_per_class, 1), sigma, generator=g)

        y0 = torch.zeros(n_samples_per_class, dtype=torch.long)
        y1 = torch.ones(n_samples_per_class, dtype=torch.long)

        self.x = torch.vstack([x0, x1])
        self.y = torch.hstack([y0, y1])
        
    def __len__(self) -> int:
        return self.x.size(0)   
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]