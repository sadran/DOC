from torch.utils.data import Dataset
import torch

class Mnist(Dataset):
    def __init__(self, images_path: str, labels_path: str, n_samples: int):
        super().__init__()
        self.images_path = images_path
        self.labels_path = labels_path
        self.x, self.y = self.load_mnist(self.images_path, self.labels_path, n_samples)
                                            
    def load_mnist(self, images_path, labels_path, n_samples):
        with open(labels_path, 'rb') as lbpath:
            lbpath.read(8)
            labels = torch.frombuffer(lbpath.read(), dtype=torch.uint8).long()

        with open(images_path, 'rb') as imgpath:
            imgpath.read(16)
            images = torch.frombuffer(imgpath.read(), dtype=torch.uint8)
            images = images.view(labels.size(0), 28*28).float() / 255.0
        
        one_images = images[labels == 1]
        two_images = images[labels == 2]
        one_lables = torch.zeros_like(labels[labels == 1])
        two_lables = torch.ones_like(labels[labels == 2])

        n_per_class = n_samples // 2

        if n_per_class > one_lables.size(0) or n_per_class > two_lables.size(0):
            raise ValueError(f"Requested {n_samples} samples, but not enough samples of each class available.")

        one_perm = torch.randperm(one_lables.size(0))[:n_per_class]
        two_perm = torch.randperm(two_lables.size(0))[:n_per_class]
        images = torch.cat([one_images[one_perm], two_images[two_perm]])
        labels = torch.cat([one_lables[one_perm], two_lables[two_perm]])
        return images, labels
        
    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]  