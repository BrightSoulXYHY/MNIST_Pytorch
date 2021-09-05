# MNIST_Pytorch

CPU是i5-7300HQ，GPU为1050Ti

对Relu来说CPU训练和GPU训练时间差不多，都是7s~8s

![relu](img/readme/image-20210905150938156.png)

但是对CNN来说GPU算的就比CPU快不少。

对于散装的jpg数据和打包成u-byte的数据读取速度是差不多的。