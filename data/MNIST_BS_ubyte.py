#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-08-21 22:54:11
# @Author  : BrightSoul (653538096@qq.com)



from PIL import Image
import os
import numpy as np
import torch
import codecs
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple


# 继承自父类torch.utils.data.Dataset
class MNIST_BS_ubyte(Dataset):   
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train

        self.data, self.targets = self._load_data()


        self.transform = transform
        self.target_transform = target_transform
 
    def _load_data(self):
        image_file = f"MNIST_BS_ubyte-image-{'train' if self.train else 'test'}"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"MNIST_BS_ubyte-label-{'train' if self.train else 'test'}"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)



def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)


SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype('>i2'), 'i2'),
    12: (torch.int32, np.dtype('>i4'), 'i4'),
    13: (torch.float32, np.dtype('>f4'), 'f4'),
    14: (torch.float64, np.dtype('>f8'), 'f8')
}



def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    # nd = 3, ty=8 -> magic = 8*256+3
    # nd = 1, ty=8 -> magic = 8*256+1
    nd = magic % 256 
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=np.uint8, offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x

if __name__ == '__main__':
    import torchvision
    test_dataset = MNIST_BS_ubyte(root='.',transform=torchvision.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    for images, labels in test_loader:
        print(type(labels))
        break