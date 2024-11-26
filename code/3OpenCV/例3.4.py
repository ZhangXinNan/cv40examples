# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:03:21 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""


import cv2
import numpy as np
img=np.zeros((8,8),dtype=np.uint8)
print("img=\n",img)
cv2.imshow("one",img)
print("读取像素点img[0,3]=",img[0,3])
img[0,3]=255
print("修改后img=\n",img)
print("读取修改后像素点img[0,3]=",img[0,3])
cv2.imshow("two",img)
cv2.waitKey()
cv2.destroyAllWindows()
