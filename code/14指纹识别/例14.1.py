# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 10:24:00 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
o=cv2.imread("lena.bmp",cv2.IMREAD_GRAYSCALE)
r1=cv2.pyrDown(o)
r2=cv2.pyrDown(r1)
r3=cv2.pyrDown(r2)
print("o.shape=",o.shape)
print("r1.shape=",r1.shape)
print("r2.shape=",r2.shape)
print("r3.shape=",r3.shape)
# cv2.imwrite("o.bmp",o)
# cv2.imwrite("r1.bmp",r1)
# cv2.imwrite("r2.bmp",r2)
# cv2.imwrite("r3.bmp",r3)
cv2.imshow("original",o)
cv2.imshow("r1",r1)
cv2.imshow("r2",r2)
cv2.imshow("r3",r3)
cv2.waitKey()
cv2.destroyAllWindows()
