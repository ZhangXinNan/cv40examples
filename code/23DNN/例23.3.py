# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 17:32:43 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
image=cv2.imread("lena.bmp")
image=cv2.resize(image,(3,3))
print("原始数据：\n",image)
blob = cv2.dnn.blobFromImage(image,1,size=(3,3),mean=0)
print("mean=0：\n",blob[0])
blob = cv2.dnn.blobFromImage(image,1,size=(3,3),mean=(10,20,50))
print("mean=(10,20,50)：\n",blob[0])
