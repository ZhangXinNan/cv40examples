# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 10:46:04 2018

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""
#图层提取
import cv2
import numpy as np
lena=cv2.imread("image\lena.bmp",0)
cv2.imshow("lena",lena)
r,c=lena.shape
x=np.zeros((r,c,8),dtype=np.uint8)
for i in range(8):
    x[:,:,i]=2**i
ri=np.zeros((r,c,8),dtype=np.uint8)
for i in range(8):
    ri[:,:,i]=cv2.bitwise_and(lena,x[:,:,i])
    mask=ri[:,:,i]>0
    ri[mask]=255
    cv2.imshow(str(i),ri[:,:,i])
cv2.waitKey()
cv2.destroyAllWindows()
