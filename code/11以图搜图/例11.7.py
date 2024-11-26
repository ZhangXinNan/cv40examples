# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 22:02:11 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""


import cv2
import numpy as np
#========提取感知哈希值函数===============
def getHash(I):
    size=(8,8)
    I=cv2.resize(I,size)
    I=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    m=np.mean(I)
    r=(I>m).astype(int)
    x=r.flatten()
    return x
#========测试感知哈希值提取函数===============
o=cv2.imread("lena.bmp")
h=getHash(o)
print("lena.bmp的感知哈希值为：\n",h)