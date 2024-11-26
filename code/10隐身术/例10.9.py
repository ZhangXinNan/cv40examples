# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:57:24 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
a=cv2.imread("boat.bmp",0)
b=cv2.imread("lena.bmp",0)
result=cv2.addWeighted(a,0.6,b,0.4,0)
cv2.imshow("boat",a)
cv2.imshow("lena",b)
cv2.imshow("result",result)
cv2.waitKey()
cv2.destroyAllWindows()
