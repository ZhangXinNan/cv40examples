# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 14:56:32 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""
#================导入库===================
import cv2
import numpy as np
#===============读取前景图像、背景图像===============
A=cv2.imread("back.jpg")
cv2.imshow('A',A)
B=cv2.imread("fore.jpg")
cv2.imshow('B',B)
#==============获取模板图像mask1/mask2===================
#转换到hsv空间，方便识别红色
hsv = cv2.cvtColor(B, cv2.COLOR_BGR2HSV)
# 红色区间1
lower_red = np.array([0,120,70])
upper_red = np.array([10,255,255])
mask1 = cv2.inRange(hsv,lower_red,upper_red)
# 红色区间2
lower_red = np.array([170,120,70])
upper_red = np.array([180,255,255])
mask2 = cv2.inRange(hsv,lower_red,upper_red)
# 模板mask1，红色整体区间 = 红色区间1+红色区间2
mask1 = mask1+mask2
cv2.imshow('mask1',mask1)
# 模板mask2，对mask1按位取反 ,黑变白、白变黑 
mask2 = cv2.bitwise_not(mask1)
cv2.imshow('mask2',mask2)
#===============图像C：背景中与前景红色斗篷对应位置图像================
C = cv2.bitwise_and(A,A,mask=mask1)
cv2.imshow('C',C)
#===============图像D：前景中抠除红色斗篷区域================
# 提取B中，mask2指定的范围
D = cv2.bitwise_and(B,B,mask=mask2)
cv2.imshow('D',D)
#===============图像E：图像C+图像D================
E=C+D
cv2.imshow('E',E)
cv2.waitKey()
cv2.destroyAllWindows()
