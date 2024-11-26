# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:53:44 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""


import cv2
a=cv2.imread("lenacolor.png",cv2.IMREAD_UNCHANGED)
face=a[220:400,250:350]
cv2.imshow("original",a)
cv2.imshow("face",face)
cv2.waitKey()
cv2.destroyAllWindows()
