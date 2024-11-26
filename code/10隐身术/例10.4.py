# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 07:00:28 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

#================导入库、读取图像===================
import cv2
B=cv2.imread("fore.jpg")
mask2=cv2.imread("mask2.bmp",0)
#================掩模控制的按位与运算===================
D = cv2.bitwise_and(B,B,mask=mask2)
#================显示图像===================
cv2.imshow('B',B)
cv2.imshow('mask2',mask2)
# cv2.imwrite("d.bmp",D)
cv2.imshow('D',D)
cv2.waitKey()
cv2.destroyAllWindows()