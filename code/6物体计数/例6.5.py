# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:00:55 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""

import cv2
o=cv2.imread("kernel.bmp",cv2.IMREAD_UNCHANGED)
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (59,59))
kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (59,59))
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (59,59))
dst1 = cv2.dilate(o,kernel1)
dst2 = cv2.dilate(o,kernel2)
dst3 = cv2.dilate(o,kernel3)
cv2.imshow("orriginal",o)
cv2.imshow("dst1",dst1)
cv2.imshow("dst2",dst2)
cv2.imshow("dst3",dst3)
cv2.waitKey()
cv2.destroyAllWindows()
