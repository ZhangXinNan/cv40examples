# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 19:50:30 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""

import cv2
img=cv2.imread("lena.bmp")
t,rst=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("img",img)
cv2.imshow("rst",rst)
cv2.waitKey()
cv2.destroyAllWindows()