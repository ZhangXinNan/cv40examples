# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 22:22:12 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import numpy as np
import cv2
#==========构造计算汉明距离函数============
def hamming(h1, h2):
    r=cv2.bitwise_xor(h1,h2)
    h=np.sum(r)
    return h
#==========使用函数计算距离============
test=np.array([0,1,1,1])
x1=np.array([0,1,1,1])
x2=np.array([1,1,1,1])
x3=np.array([1,0,0,0])
t1=hamming(test,x1)
t2=hamming(test,x2)
t3=hamming(test,x3)
print("test(0111)和x1(0111)的距离：",t1)
print("test(0111)和x2(1111)的距离：",t2)
print("test(0111)和x3(1000)的距离：",t3)
