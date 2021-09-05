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
import numpy as np
import cv2

'''
采集手写数据的脚本
'''

save_path = "img"


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

        self.save_count = {str(i):0 for i in range(10)}

        super(MyMnistWindow, self).__init__()

        # resize设置宽高
        self.resize(310, 350)  
 
        
        # self.setWindowFlags(Qt.FramelessWindowHint)  # 窗体无边框
        self.setWindowFlags(Qt.MSWindowsFixedSizeDialogHint)
        
        self.title = "MNIST-Get"
        self.setWindowTitle(self.title)
        #setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        self.setMouseTracking(False)

  

        # 添加一系列控件
        self.label_info = QLabel('数字', self)
        # self.label_info.setStyleSheet("QLabel{border:1px solid black;}")
        # self.label_info.setAlignment(Qt.AlignCenter)
        
        self.cb = QComboBox(self)
        self.cb.addItems([str(i) for i in range(10)]) 
        
        self.btn_save = QPushButton("保存并清空", self)
        self.btn_clr = QPushButton("清空", self)
        


        top_hbox = QHBoxLayout()
        top_hbox.addWidget(self.label_info,0,Qt.AlignCenter)
        top_hbox.addWidget(self.cb)
        top_hbox.addWidget(self.btn_save)
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

        vbox = QVBoxLayout()

        vbox.addLayout(top_hbox)
        vbox.addWidget(draw_group)
        # vbox.addWidget(self.label_draw)
        # vbox.addLayout(self.label_draw)


        self.setLayout(vbox) 


        self.btn_save.clicked.connect(self.btn_save_on_clicked)
        self.btn_clr.clicked.connect(self.btn_clr_on_clicked)


    def btn_clr_on_clicked(self):
        self.label_draw.clear_draw()
    
    def btn_save_on_clicked(self):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            print(f"make dir {save_path}")


        cnt_num = self.cb.currentText()
        cnt_save = self.save_count[cnt_num]

        img_name = f"{save_path}/{cnt_num}_{cnt_save}.jpg"

        screen = QApplication.primaryScreen()
        screenshot = screen.grabWindow( self.label_draw.winId() )
        screenshot.save(img_name, 'jpg')

        self.save_count[cnt_num] += 1

        print(img_name,"save done")
        self.label_draw.clear_draw()
 
    # def btn_save_on_clicked(self):
    #     screen = QApplication.primaryScreen()
    #     screenshot = screen.grabWindow( self.label_draw.winId() )
        
    #     qimg = screenshot.toImage()
    #     qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
        
    #     width = qimg.width()
    #     height = qimg.height()

    #     ptr = qimg.bits()
    #     ptr.setsize(height * width * 3)
    #     img = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    #     th, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    #     img = cv2.bitwise_not(img)
    #     img = cv2.resize(img,(28, 28))

    #     cv2.imshow("ss",img)
    #     cv2.waitKey(0)




   

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mymnist = MyMnistWindow()
    mymnist.show()
    app.exec_()
