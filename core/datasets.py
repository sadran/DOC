from torch.utils.data import Dataset
import torch
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms

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

        mu0[0] = mean_distance / 2
        mu1[0] = -mean_distance / 2

        x0 = torch.normal(mu0.repeat([n_samples_per_class, 1]), sigma)
        x1 = torch.normal(mu1.repeat([n_samples_per_class, 1]), sigma)

        y0 = torch.zeros(n_samples_per_class, dtype=torch.long)
        y1 = torch.ones(n_samples_per_class, dtype=torch.long)

        self.x = torch.vstack([x0, x1])
        self.y = torch.hstack([y0, y1])
        
    def __len__(self) -> int:
        return self.x.size(0)   
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


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


class ImageNet1k(Dataset):
    def __init__(self, data_root_dir: str, split: str = "train", n_samples:int = 1000):
        """
        this is a sub-set of ImageNet1k dataset including 1300 images of 'goldfish' class 
        and 1300 images of 'airliner' class.

        :param data_root_dir: path to the data root directory
        :type data_root_dir: str

        :param split: train / test.
            it draws '# n_samples' samples from both classes balancedly according to the split.
        :type split: str
        :param n_samples: Description
        :type n_samples: int
        """
        super().__init__()
        image_folder = ImageFolder(data_root_dir)
        self.classes = image_folder.classes
        self.class_to_idx = image_folder.class_to_idx
        self.split = split
        # draw samples according to the split
        self.samples = self.__draw_samples(image_folder, n_samples)
        # transforms
        self.transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                   std=(0.229, 0.224, 0.225)),])
        
        self.x = torch.stack([self.transforms(Image.open(path).convert("RGB")) for path, _ in self.samples])
        self.y = torch.tensor([target for _, target in self.samples], dtype=torch.long)

    def __draw_samples(self,data: ImageFolder, n_samples: int):
        """
        description: draws samples from the given ImageFolder dataset according to the split (train/test) and the number of samples requested.
        :param data: ImageFolder dataset object containing the samples to draw from
        :type data: ImageFolder
        """
        # filtering samples by class
        class_data = {cls: [] for cls in self.classes}
        for path, target in data.samples:
            class_data[self.classes[target]].append((path, target))

        # sampling reandomly from each class
        n_per_class = n_samples // 2
        if self.split == "train":
            # for train split, we draw the samples from begining of the class data (first 500 samples for each class)
            samples = []
            for cls, data in class_data.items():
                perm = torch.randperm(n_per_class)
                samples.extend([data[i] for i in perm])
        elif self.split == "test":
            # for test split, we draw the samples from the end of the class data (last 500 samples for each class)
            samples = []
            for cls, data in class_data.items():
                perm = torch.randperm(n_per_class)
                samples.extend([data[-i] for i in perm])
        else:
            raise ValueError(f"Invalid split: {self.split}. Expected 'train' or 'test'.")    
        return samples  


    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")
            if self.transforms is not None:
                img = self.transforms(img)
        return img, target
    
