import os
import torch
import torchvision

import time
from tqdm import tqdm

from data.MNIST_BS_jpg import MNIST_BS_jpg

from NN_CNN import CNN_MNIST

# from data.BS_MNIST import BS_MNIST

data_root = './data'

# 设置超参数
batch_size = 100
num_classes = 10
num_epochs = 500
learning_rate = 0.01
# learning_rate = 0.001


model_path = 'weights_log/MNIST_CNN_device=cuda-epoch=10-acc=83.33.pth'
# 保存相关参数
# save_text.format(device, epoch + 1 ,accuracy)
save_text = 'weights_log/MNIST_CNN_device={}-epoch={:02d}-acc={:.2f}.pth'
continue_train = True


# CUDA相关参数
cuda = True
# cuda = False
cuda_enable = cuda and torch.cuda.is_available()


# 从TorchVision下载MNIST数据集
# train_dataset = BS_MNIST(root='./data', train=True,transform=torchvision.transforms.ToTensor())
# test_dataset = BS_MNIST(root='./data', train=False,transform=torchvision.transforms.ToTensor())

train_dataset = MNIST_BS_jpg(root=data_root, train=True, transform=torchvision.transforms.ToTensor())
test_dataset = MNIST_BS_jpg(root=data_root, train=False, transform=torchvision.transforms.ToTensor())


# 使用PyTorch提供的DataLoader，以分批乱序加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size*6, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)




# 实例化模型
model = CNN_MNIST()

# 若cuda可用则转化为并行网络
device = torch.device('cuda' if cuda_enable else 'cpu')
if cuda_enable:
    model = torch.nn.DataParallel(model)
    model = model.cuda()

# 断点训练
if continue_train and os.path.exists(model_path):
    if cuda_enable:
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # CPU加载GPU的训练的参数
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})


# 设置损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



start_time = time.time()
print('[{:.2f}]'.format(time.time()-start_time),"start train")
# 训练模型
for epoch in range(num_epochs):
    total_loss = 0
    val_loss = 0

    # 训练的进度条
    with tqdm(total=100,desc=f'Train {epoch + 1}/{num_epochs}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            t1 = time.time()
            # 将图片从28*28的矩阵拉成784的向量
            images,labels = batch 
            if cuda:
                images,labels = images.cuda(),labels.cuda()



            # 前向传播获得模型的预测值
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播算出Loss对各参数的梯度
            optimizer.zero_grad()
            loss.backward()

            # 更新参数
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1)})
            pbar.update(1)


    
    # 检验模型在测试集上的准确性
    with tqdm(desc=f'Test {epoch + 1}/{num_epochs}',postfix=dict,mininterval=0.3) as pbar:
        
        correct = 0
        total = 0
        for images, labels in test_loader:
            if cuda:
                images,labels = images.cuda(),labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            pbar.set_postfix(**{'accuracy': accuracy})
            pbar.update(1)

    if accuracy >= 98.5:
        torch.save(model.state_dict(), save_text.format(device, epoch + 1 ,accuracy) )
        break

    if not (epoch + 1) % 10:
        torch.save(model.state_dict(), save_text.format(device, epoch + 1 ,accuracy) )

