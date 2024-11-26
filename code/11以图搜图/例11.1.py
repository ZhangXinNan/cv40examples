# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 17:34:39 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
img=cv2.imread("lena.bmp")
size=(8,8)
rst=cv2.resize(img,size)
# cv2.imwrite("rst.bmp",rst)
print("img.shape=",img.shape)
print("rst.shape=",rst.shape)
