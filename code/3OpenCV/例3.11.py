# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 17:31:44 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""

import cv2
import numpy as np
m1=np.zeros([600,600],np.uint8)
m1[200:400,200:400]=255
m2=np.zeros([600,600],np.uint8)
m2[200:400,200:400]=1
cv2.imshow('m1',m1)
cv2.imshow('m2',m2)
cv2.imshow('m2*255',m2*255)
cv2.waitKey()
cv2.destroyAllWindows()
