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
        
        self.x = torch.empty((0, 3, 32, 32), dtype=torch.float32)
        self.y = torch.empty((0,), dtype=torch.long)

        self.x = torch.cat([self.x, torch.tensor(dataset.data[torch.tensor(dataset.targets) == 0]).permute(0, 3, 1, 2).float() / 255.0], dim=0)
        self.y = torch.cat([self.y, torch.zeros((self.x.shape[0],), dtype=torch.long)], dim=0)
        self.x = torch.cat([self.x, torch.tensor(dataset.data[torch.tensor(dataset.targets) == 5]).permute(0, 3, 1, 2).float() / 255.0], dim=0)
        self.y = torch.cat([self.y, torch.ones((self.x.shape[0] - self.y.shape[0],), dtype=torch.long)], dim=0)
        
        if num_samples is not None and num_samples < len(self.y):
            perm = torch.randperm(len(self.y))[:num_samples]
            self.x = self.x[perm]
            self.y = self.y[perm]

    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        return self.x[index], self.y[index]
    