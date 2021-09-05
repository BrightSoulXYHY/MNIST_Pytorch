import os
import torch
import torchvision

import time
from tqdm import tqdm

from NN_Relu import Relu_MNIST

from data.MNIST_BS_jpg import MNIST_BS_jpg
from data.MNIST_BS_ubyte import MNIST_BS_ubyte

start_time = time.time()

data_root = './data'

# 设置超参数
batch_size = 100
input_size = 784
hidden_size = 1000
num_classes = 10


model_path = 'weights/Relu-epoch-14-100.00.pth'
# model_path = 'weights_log/MNIST_Relu_device=cuda-epoch=01-acc=97.73.pth'

# CUDA相关参数
cuda = True
# cuda = False
cuda_enable = cuda and torch.cuda.is_available()


# 从TorchVision下载MNIST数据集
# test_dataset = torchvision.datasets.MNIST(root=data_root, train=False, transform=torchvision.transforms.ToTensor())

test_dataset = MNIST_BS_jpg(root=data_root, train=False, transform=torchvision.transforms.ToTensor())


test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)





# 实例化模型（可调用对象）
model = Relu_MNIST(input_size, hidden_size, num_classes)


# 要先转换为并行的
# 若cuda可用则转化为并行网络
if cuda_enable:
    model = torch.nn.DataParallel(model)
    model = model.cuda()

device = torch.device('cuda' if cuda_enable else 'cpu')

# 加载参数
if os.path.exists(model_path):
    if cuda_enable:
        # GPU加载GPU参数
        model.load_state_dict(torch.load(model_path, map_location=device))
        # GPU加载CPU参数
        # model.load_state_dict({f'module.{k}': v for k, v in torch.load(model_path).items()})
    else:
        # CPU加载GPU的训练的参数或者CPU加载CPU参数
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
    print("Load state_dict done")




print('[{:.2f}]'.format(time.time()-start_time),"start test")



with tqdm(postfix=dict,mininterval=0.3) as pbar:
    correct = 0
    total = 0
    for images, labels in test_loader:
        images,labels = images.reshape(-1, 28 * 28),labels

        if cuda:
            images,labels = images.cuda(),labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        pbar.set_postfix(**{'accuracy': accuracy})
        pbar.update(1)

# with tqdm(postfix=dict,mininterval=0.3) as pbar:
    
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         if cuda:
#             images,labels = images.cuda(),labels.cuda()
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#         pbar.set_postfix(**{'accuracy': 100 * correct / total})
#         pbar.update(1)
