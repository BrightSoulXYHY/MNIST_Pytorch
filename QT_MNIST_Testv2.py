#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-08-19 16:09:13
# @Author  : BrightSoul (653538096@qq.com)


from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QComboBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGroupBox
from PyQt5.QtGui import QPainter, QPen, QFont, QImage, QPixmap
from PyQt5.QtCore import Qt,QRect


import sys,os
from PyQt5.QtWidgets import QApplication


import time


import torch
import torchvision
from NN_CNN import CNN_MNIST
from NN_Relu import Relu_MNIST

import numpy as np
import cv2



model_relu = "weights/MNIST_BS_Relu_device=cuda-epoch=150-acc=75.00.pth"
model_cnn = "weights/MNIST_BS_CNN_device=cuda-epoch=500-acc=80.00.pth"
# model_cnn = "weights/CNN-MNIST-epoch=30-acc=99.04.pth"
cuda = False

cuda_enable = torch.cuda.is_available() and cuda

class DrawLabel(QLabel):
    def __init__(self, parent=None):
        super(DrawLabel, self).__init__((parent))
        self.pos_xy = []  #保存鼠标移动过的点
 
    def paintEvent(self, event):
        super().paintEvent(event)
        painter=QPainter(self)
        pen = QPen(Qt.black, 22, Qt.SolidLine)
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()


    def mouseMoveEvent(self, event):
        '''
            按住鼠标移动事件：将当前点添加到pos_xy列表中
        '''
        #中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        #pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        '''
            重写鼠标按住后松开的事件
            在每次松开后向pos_xy列表中添加一个断点(-1, -1)
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()

    def clear_draw(self):
        self.pos_xy = []
        self.update()


class MyMnistWindow(QWidget):

    def __init__(self):
        self.init_nn()


        super(MyMnistWindow, self).__init__()

        # resize设置宽高
        self.resize(310, 350)  
 
        
        # self.setWindowFlags(Qt.FramelessWindowHint)  # 窗体无边框
        self.setWindowFlags(Qt.MSWindowsFixedSizeDialogHint)
        
        self.title = "MNIST-Test"
        self.setWindowTitle(self.title)
        #setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        self.setMouseTracking(False)

  

        # 添加一系列控件
        self.label_info = QLabel('模型', self)
        # self.label_info.setStyleSheet("QLabel{border:1px solid black;}")
        # self.label_info.setAlignment(Qt.AlignCenter)
        
        self.cb = QComboBox(self)
        self.cb.addItems(["CNN","Relu"]) 
        
        self.btn_reco = QPushButton("识别", self)
        self.btn_clr = QPushButton("清空", self)
        


        top_hbox = QHBoxLayout()
        top_hbox.addWidget(self.label_info,0,Qt.AlignCenter)
        top_hbox.addWidget(self.cb)
        top_hbox.addWidget(self.btn_reco)
        top_hbox.addWidget(self.btn_clr)





        self.label_draw = DrawLabel(self)
        # vbox.setStyleSheet("border:1px solid black")
        self.label_draw.setMinimumWidth(280)
        self.label_draw.setMinimumHeight(280)

        draw_hbox = QHBoxLayout()
        draw_hbox.addWidget(self.label_draw,0,Qt.AlignCenter)

        draw_group = QGroupBox(self)
        draw_group.setLayout(draw_hbox)
        draw_group.setStyleSheet("color:black")



        self.label_result = QLabel('', self)


        vbox = QVBoxLayout()

        vbox.addLayout(top_hbox)
        vbox.addWidget(draw_group)
        vbox.addWidget(self.label_result,0,Qt.AlignCenter)
        # vbox.addWidget(self.label_draw)



        self.setLayout(vbox) 


        self.btn_reco.clicked.connect(self.btn_reco_on_clicked)
        self.btn_clr.clicked.connect(self.btn_clr_on_clicked)


    def init_nn(self):
        if not (os.path.exists(model_relu) and os.path.exists(model_relu)):
            print("model_path is invalid")
            exit()


        self.device = torch.device('cuda' if cuda_enable else 'cpu')

        self.relu = self.init_nn_relu()
        self.cnn = self.init_nn_cnn()

    def init_nn_relu(self):
        input_size = 784
        hidden_size = 1000
        num_classes = 10

        model = Relu_MNIST(input_size, hidden_size, num_classes)
        return self.load_weights(model,model_relu)

    def init_nn_cnn(self):
        model = CNN_MNIST()
        return self.load_weights(model,model_cnn)

    def load_weights(self,model,model_path):
        if cuda_enable:
            # GPU加载GPU的参数
            model = torch.nn.DataParallel(model)
            model = model.cuda()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            # CPU加载GPU的训练的参数
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
        return model


    def btn_clr_on_clicked(self):
        self.label_draw.clear_draw()
    
    def btn_reco_on_clicked(self):
        def img_to_tensor(image):
            loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) 
            image = loader(image).unsqueeze(0)
            return image.to(self.device, torch.float)

        def recognize_img(model,im_tensor): 
            outputs = model(im_tensor)
            _, predicted = torch.max(outputs, 1)       
            return predicted.item()

        screen = QApplication.primaryScreen()
        screenshot = screen.grabWindow( self.label_draw.winId() )


        qimg = screenshot.toImage()
        qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
        
        width = qimg.width()
        height = qimg.height()

        ptr = qimg.bits()
        ptr.setsize(height * width * 3)
        img = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        th, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
        img = cv2.bitwise_not(img)
        img = cv2.resize(img,(28, 28))

        cnt_model = self.cb.currentText()
        # print(cnt_module)
        im_tensor = img_to_tensor(img)

        cnn_result = recognize_img(self.cnn,im_tensor)
        relu_result = recognize_img(self.relu,im_tensor.reshape(-1, 28 * 28))

        # if cnt_model == "CNN":
        #     outputs = self.cnn(im_tensor)
        # elif cnt_model == "Relu":
        #     images = im_tensor.reshape(-1, 28 * 28)
        #     outputs = self.relu(images)
        
        recognize_result = "CNN:{}\tRelu:{}"
        self.label_result.setText(recognize_result.format(cnn_result,relu_result))  # 显示识别结果
        self.update()






   

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mymnist = MyMnistWindow()
    mymnist.show()
    app.exec_()
