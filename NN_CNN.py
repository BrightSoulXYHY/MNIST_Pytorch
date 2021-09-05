import torch

# 构建卷积神经网络
class CNN_MNIST(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_MNIST, self).__init__()
        # 输入的数据为1x1x28x28,卷积后为1x16x28x28
        # 池化后为1x16x14x14
        self.conv_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # 继续卷积和池化，输出1x32x7x7
        self.conv_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # 全连接层进行分类
        self.fc = torch.nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)

        # 将卷积层的结果拉成向量再通过全连接层
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x