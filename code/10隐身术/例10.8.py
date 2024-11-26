# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:53:37 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
import numpy as np
img1=np.random.randint(0,256,(3,4),dtype=np.uint8)
img2=np.random.randint(0,256,(3,4),dtype=np.uint8)
img3=np.zeros((3,4),dtype=np.uint8)
gamma=3
img3=cv2.addWeighted(img1,2,img2,1,gamma)
print("img1=")
print(img1)
print("img2=")
print(img2)
print("img3=")
print(img3) 
