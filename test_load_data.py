import time

import torch
import torchvision

from data.MNIST_BS_ubyte import MNIST_BS_ubyte
from data.MNIST_BS_jpg import MNIST_BS_jpg

data_root = './data'

batch_size = 100

start_time = time.time()
bs_dataset_ubyte = MNIST_BS_ubyte(root=data_root, train=True,transform=torchvision.transforms.ToTensor())
bs_dataset_ubyte_loader = torch.utils.data.DataLoader(dataset=bs_dataset_ubyte, batch_size=batch_size, shuffle=True)



print("load bs_dataset_ubyte in  {:.4f}s".format(time.time()-start_time))

start_time = time.time()
bs_dataset_jpg = MNIST_BS_jpg(root=data_root, train=True,transform=torchvision.transforms.ToTensor())
bs_dataset_jpg_loader = torch.utils.data.DataLoader(dataset=bs_dataset_ubyte, batch_size=batch_size, shuffle=True)
print("load bs_dataset_jpg in  {:.4f}s".format(time.time()-start_time))
