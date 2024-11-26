# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 19:14:39 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import numpy as np
import cv2
img=cv2.imread("rst88.bmp",-1)
print("img=\n",img)
m=np.mean(img)
print("平均值：",m)
r=img>m
print("特征值:\n",r.astype(int))