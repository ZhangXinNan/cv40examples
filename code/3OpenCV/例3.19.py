# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 20:49:20 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""

import cv2
import numpy as np
o=cv2.imread("erode.bmp",cv2.IMREAD_UNCHANGED)
kernel1 = np.ones((5,5),np.uint8)
erosion1 = cv2.erode(o,kernel1)
kernel2 = np.ones((9,9),np.uint8)
erosion2 = cv2.erode(o,kernel2,iterations = 5)
cv2.imshow("orriginal",o)
cv2.imshow("erosion1",erosion1)
cv2.imshow("erosion2",erosion2)
cv2.waitKey()
cv2.destroyAllWindows()
