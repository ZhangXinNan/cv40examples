# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 20:06:20 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""
import cv2
import numpy as np
img=cv2.imread("x.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
min_HSV = np.array([0 ,10,80], dtype = "uint8")
max_HSV = np.array([33, 255, 255], dtype = "uint8")
mask = cv2.inRange(hsv, min_HSV, max_HSV)
reusult = cv2.bitwise_and(img,img, mask= mask)
cv2.imshow("img",img)
cv2.imshow("reusult",reusult)
cv2.waitKey()
cv2.destroyAllWindows()