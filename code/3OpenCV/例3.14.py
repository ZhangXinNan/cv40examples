# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 17:36:08 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""

import cv2
import numpy  as np
o=cv2.imread("lenacolor.png",1) 
t=cv2.imread("text.png",1) 
h,w,c=o.shape
m=np.zeros((h,w),dtype=np.uint8)
m[100:400,200:400]=255
m[100:500,100:200]=255
r=cv2.add(o,t,mask=m)
cv2.imshow("orignal",o)
cv2.imshow("text",t)
cv2.imshow("mask",m)
cv2.imshow("result",r)
cv2.waitKey()
cv2.destroyAllWindows()
