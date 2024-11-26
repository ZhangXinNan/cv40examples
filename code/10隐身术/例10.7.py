# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:30:52 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
a=cv2.imread("lena.bmp",0)
b=a
result1=a+b
result2=cv2.add(a,b)
cv2.imshow("original",a)
cv2.imshow("result1",result1)
cv2.imshow("result2",result2)
cv2.waitKey()
cv2.destroyAllWindows()
