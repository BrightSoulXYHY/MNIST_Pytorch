#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-08-21 21:33:28
# @Author  : BrightSoul (653538096@qq.com)


import os

import struct
import cv2


train = False

imgDirName = "MNIST_BS_jpg"
L = os.listdir(imgDirName)

if train:
    imgL = list(filter(lambda name: name.split('_')[0] != "YSJ", L))
    mode = "train"  
else:
    imgL = list(filter(lambda name: name.split('_')[0] == "YSJ", L))
    mode = "test"  


numImages = len(imgL)

img_bin_name    =   f"MNIST_BS_ubyte-image-{mode}"
label_bin_name  =   f"MNIST_BS_ubyte-label-{mode}"


numRows, numColumns = 28,28
imgSize = numRows*numColumns
# nd = 3, ty=8 -> magic = 8*256+3
img_head = struct.pack( '>4I', 8*256+3, numImages, numRows, numColumns)
label_head = struct.pack( '>2I', 8*256+1, numImages)


img_bin = open(img_bin_name,"wb")
label_bin = open(label_bin_name,"wb")

img_bin.write(img_head)
label_bin.write(label_head)


for imgName in imgL:
    img = cv2.imread(os.path.join(imgDirName,imgName),cv2.IMREAD_GRAYSCALE)
    label = int(imgName.split("_")[1])

    img_bin.write(struct.pack(f">{imgSize}B",*img.ravel()))
    label_bin.write(struct.pack(">B",label))

img_bin.close()
label_bin.close()