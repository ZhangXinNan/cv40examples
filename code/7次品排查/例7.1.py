# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 20:02:49 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""

import cv2
import numpy as np
img1=cv2.imread("opening.bmp")
img2=cv2.imread("opening2.bmp")
k=np.ones((10,10),np.uint8)
r1=cv2.morphologyEx(img1,cv2.MORPH_OPEN,k)
r2=cv2.morphologyEx(img2,cv2.MORPH_OPEN,k)
cv2.imshow("img1",img1)
cv2.imshow("result1",r1)
cv2.imshow("img2",img2)
cv2.imshow("result2",r2)
cv2.waitKey()
cv2.destroyAllWindows()
