# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 17:32:43 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import numpy as np
import cv2
image=np.ones((3,3),np.uint8)*100
print("原始数据：\n",image)
print("------------ddepth=CV_32F------------")
blob = cv2.dnn.blobFromImage(image,1,(3,3),0,False,False,ddepth=cv2.CV_32F)
print("blob数据类型：",blob.dtype)
print("观察一下值：\n",blob[0])
print("------------ddepth=CV_8U------------")
blob = cv2.dnn.blobFromImage(image,1,(3,3),0,False,False,ddepth=cv2.CV_8U)
print("blob数据类型：",blob.dtype)
print("观察一下值：\n",blob[0])