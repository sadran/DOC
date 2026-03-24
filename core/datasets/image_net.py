from torch.utils.data import Dataset
import torch
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms



class ImageNet1k(Dataset):
    def __init__(self, data_root_dir: str, split: str = "train", n_samples:int = 1000):
        """
        this is a sub-set of ImageNet1k dataset including 1300 images of 'goldfish' class 
        and 1300 images of 'airliner' class. image size is 224x224.

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
    