# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:58:06 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""


import cv2
lena=cv2.imread("lena.bmp")
cv2.imshow("demo1", lena )
cv2.imshow("demo2", lena )
cv2.waitKey()
cv2.destroyAllWindows()
