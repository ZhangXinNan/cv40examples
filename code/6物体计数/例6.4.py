# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:00:26 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社 
"""

import cv2
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,  (5,5))
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (5,5))
print("kernel1=\n",kernel1)
print("kernel2=\n",kernel2)
print("kernel3=\n",kernel3)
