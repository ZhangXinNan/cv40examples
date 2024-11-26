# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 18:38:00 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
img=cv2.imread("rst.bmp")
rst=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print("img.shape=",img.shape)
print("rst.shape=",rst.shape)
# cv2.imwrite("rst88.bmp",rst)