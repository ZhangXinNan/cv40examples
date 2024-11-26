# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:44:44 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""


import cv2
lena=cv2.imread("lenacolor.png")
b,g,r=cv2.split(lena)
bgr=cv2.merge([b,g,r])
rgb=cv2.merge([r,g,b])
cv2.imshow("lena",lena)
cv2.imshow("bgr",bgr)
cv2.imshow("rgb",rgb)
cv2.waitKey()
cv2.destroyAllWindows()
