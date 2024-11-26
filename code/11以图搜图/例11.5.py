# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 19:23:48 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import numpy as np
import cv2
img=cv2.imread("rst88.bmp",-1)
m=np.mean(img)
r=(img>m).astype(int)
print("特征值:\n",r)
r1=r.reshape(-1)
r2=r.ravel()
r3=r.flatten()
print("r1=\n",r1)
print("r2=\n",r2)
print("r3=\n",r3)

