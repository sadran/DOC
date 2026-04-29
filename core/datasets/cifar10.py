from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import transforms

class Cifar10(Dataset):
    def __init__(self, root='data/cifar-10-python', train: bool="False", num_samples: int = None):
        super().__init__()
        
        self.transform = transforms.Compose([ transforms.ToTensor(),  # convert to tensor
                                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                  (0.2023, 0.1994, 0.2010)) # normalize
                                             ])
        
        dataset = torchvision.datasets.CIFAR10(root=root, 
                                               train=train,
                                               download=False,
                                               transform=self.transform)
        
        self.X = torch.empty((0, 3, 32, 32), dtype=torch.float32)
        self.Y = torch.empty((0,), dtype=torch.long)

        self.X = torch.cat([self.X, torch.tensor(dataset.data[torch.tensor(dataset.targets) == 0]).permute(0, 3, 1, 2).float() / 255.0], dim=0) # airplane
        self.Y = torch.cat([self.Y, torch.zeros((self.X.shape[0] - self.Y.shape[0],), dtype=torch.long)], dim=0)
        self.X = torch.cat([self.X, torch.tensor(dataset.data[torch.tensor(dataset.targets) == 5]).permute(0, 3, 1, 2).float() / 255.0], dim=0) # dog
        self.Y = torch.cat([self.Y, torch.ones((self.X.shape[0] - self.Y.shape[0],), dtype=torch.long)], dim=0)

        if num_samples is not None and num_samples < len(self.Y):
            perm = torch.randperm(len(self.Y))[:num_samples]
            self.x = self.X[perm]
            self.y = self.Y[perm]
        else:
            self.x = self.X
            self.y = self.Y

    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]
    
    def sample_new_data(self, num_samples: int):
        """Resample a new set of data points from the dataset balanced."""
        perm0 = torch.randperm((self.Y == 0).sum())[:num_samples // 2]
        perm1 = torch.randperm((self.Y == 1).sum())[:num_samples // 2]
        self.x = torch.cat([self.X[self.Y == 0][perm0], self.X[self.Y == 1][perm1]], dim=0)
        self.y = torch.cat([self.Y[self.Y == 0][perm0], self.Y[self.Y == 1][perm1]], dim=0)