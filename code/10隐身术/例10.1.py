# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 16:52:53 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

#================导入库===================
import cv2
import numpy as np
#===============读取前景图像、背景图像===============
B=cv2.imread("fore.jpg")
cv2.imshow('B',B)
#==============获取模板图像mask1===================
#转换到hsv空间，方便识别红色
hsv = cv2.cvtColor(B, cv2.COLOR_BGR2HSV)
# 红色区间1(maskA)
redLower = np.array([0,100,100])
redUpper = np.array([10,255,255])
maskA = cv2.inRange(hsv,redLower,redUpper)
# 红色区间2(maskB)
redLower = np.array([160,100,100])
redUpper = np.array([179,255,255])
maskB = cv2.inRange(hsv,redLower,redUpper)
# 红色整体区间 = 红色区间1+红色区间2
mask1 = maskA+maskB
#==============显示图像mask1===================
cv2.imshow('mask1',mask1)
cv2.imwrite("mask1.bmp",mask1)
cv2.waitKey()
cv2.destroyAllWindows()
