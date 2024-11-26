# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 21:29:25 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

#================导入库、读取图像===================
import cv2
mask1=cv2.imread("mask1.bmp",0)
#================获取mask1的反色图像===================
mask2a = cv2.bitwise_not(mask1)   #方式A：按位取反
mask2b = 255-mask1                #方式B：255-逐个像素
t,mask2c = cv2.threshold(mask1,127,255,cv2.THRESH_BINARY_INV)  #方式C：反二值化阈值处理
#================显示图像===================
cv2.imshow('mask1',mask1)
cv2.imshow('mask2a',mask2a)
# cv2.imwrite("mask2.bmp",mask2a)
cv2.imshow('mask2b',mask2b)
cv2.imshow('mask2c',mask2c)
cv2.waitKey()
cv2.destroyAllWindows()