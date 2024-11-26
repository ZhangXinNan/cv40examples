# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:39:16 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""

import cv2
import numpy as np
# ==============步骤1：图像预处理======================
img = cv2.imread('coins.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# ============步骤2：使用函数distanceTransform()完成距离的计算=================
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ============	步骤3：使用函数threshold()，获取确定前景。=====================
ret, fore = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# ========================步骤4：显示处理结果======================
cv2.imshow('img',img)
cv2.imshow('fore',fore)
cv2.waitKey()
cv2.destroyAllWindows()