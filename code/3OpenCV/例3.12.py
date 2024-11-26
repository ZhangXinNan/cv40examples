# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 17:33:55 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""

import cv2
import numpy  as np
o=cv2.imread("lenacolor.png",1) 
h,w,c=o.shape
m=np.zeros((h,w,c),dtype=np.uint8)
m[100:400,200:400]=1
m[100:500,100:200]=1
result=m*o
cv2.imshow("o",o)
cv2.imshow("mask",m*255)   #m*255，确保能显示
cv2.imshow("result",result)
cv2.waitKey()
cv2.destroyAllWindows()
