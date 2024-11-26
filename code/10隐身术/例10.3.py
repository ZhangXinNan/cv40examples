# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 16:07:19 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

#================导入库、读取图像===================
import cv2
A=cv2.imread("back.jpg")
mask1=cv2.imread("mask1.bmp",0)
#================掩模控制的按位与运算===================
C = cv2.bitwise_and(A,A,mask=mask1)
#================显示图像===================
cv2.imshow('A',A)
cv2.imshow('mask1',mask1)
cv2.imshow('C',C)
# cv2.imwrite("c.bmp",C)
cv2.waitKey()
cv2.destroyAllWindows()
