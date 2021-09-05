#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-08-21 20:56:23
# @Author  : BrightSoul (653538096@qq.com)


import os
import cv2

path = "MNIST_BS"
output = "out"


def img_proc(img_path):
    prefix = os.path.basename(os.path.dirname(img_path))
    name = os.path.splitext(os.path.basename(img_path))[0] 
    out_path = os.path.join(output,f"{prefix}_{name}.jpg")

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, ret = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    img_bin = cv2.bitwise_not(ret)
    img_bin = cv2.resize(img_bin,(28, 28))
    cv2.imwrite(out_path,img_bin)


for root, dirs, files in os.walk(path):
    if len(files) == 0:
        continue
    for img in files:
        img_proc(os.path.join(root,img))
        # break 
        # print(os.path.join(root,img))
        