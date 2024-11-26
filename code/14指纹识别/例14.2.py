# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 10:50:12 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
o=cv2.imread("lena.bmp")
r1=cv2.GaussianBlur(o,(3,3),0,0)
r2=cv2.GaussianBlur(o,(13,13),0,0)
r3=cv2.GaussianBlur(o,(21,21),0,0)
cv2.imwrite("gaussian\\o.bmp",o)
cv2.imwrite(r"gaussian\r1.bmp",r1)
cv2.imwrite(r"gaussian/r2.bmp",r2)
cv2.imwrite(r"gaussian/r3.bmp",r3)
cv2.imshow("original",o)
cv2.imshow("result1",r1)
cv2.imshow("result2",r2)
cv2.imshow("result3",r3)
cv2.waitKey()
cv2.destroyAllWindows()
