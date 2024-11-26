# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 21:08:49 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""

import cv2
import numpy as np
o=cv2.imread("dilation.bmp",cv2.IMREAD_UNCHANGED)
kernel = np.ones((5,5),np.uint8)
dilation1 = cv2.dilate(o,kernel)
dilation2 = cv2.dilate(o,kernel,iterations = 9)
cv2.imshow("original",o)
cv2.imshow("dilation1",dilation1)
cv2.imshow("dilation2",dilation2)
cv2.waitKey()
cv2.destroyAllWindows()
