from PIL import Image
import os
import numpy as np
import torch
import codecs
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple


# 继承自父类torch.utils.data.Dataset
class MNIST_BS_jpg(Dataset):   
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train

        L = os.listdir(self.raw_folder)
        if self.train:
            self.img_nameL = list(filter(lambda name: name.split('_')[0] != "YSJ", L))    
        else:
            self.img_nameL = list(filter(lambda name: name.split('_')[0] == "YSJ", L))


        self.transform = transform
        self.target_transform = target_transform




    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_name = self.img_nameL[index]
        img = Image.open(os.path.join(self.raw_folder, img_name)) 
        target = int(img_name.split("_")[1])



        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.img_nameL)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)


if __name__ == '__main__':
    import torchvision
    test_dataset = MNIST_BS_jpg(root='.',train=False,transform=torchvision.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    for images, labels in test_loader:
        # print(type(labels),)
        print(images.shape)
        break