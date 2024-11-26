# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 14:29:16 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）(待定名称)
李立宗 著     电子工业出版社
"""

import cv2
o = cv2.imread('contours.bmp')  
gray = cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
contours, hierarchy = cv2.findContours(binary,
                                             cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)  
hull = cv2.convexHull(contours[0])   # 返回坐标值
print("returnPoints为默认值True时返回值hull的值：\n",hull)
hull2 = cv2.convexHull(contours[0], returnPoints=False) # 返回索引值
print("returnPoints为False时返回值hull的值：\n",hull2)
