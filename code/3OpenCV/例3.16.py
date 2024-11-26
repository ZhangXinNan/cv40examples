# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 20:18:57 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""

import cv2
o=cv2.imread("lenaNoise.png")
r3=cv2.blur(o,(3,3))      
r11=cv2.blur(o,(11,11))      
cv2.imshow("original",o)
cv2.imshow("result3",r3)
cv2.imshow("result11",r11)
cv2.waitKey()
cv2.destroyAllWindows()
