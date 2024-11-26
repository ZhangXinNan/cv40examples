# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 14:31:21 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）(待定名称)
李立宗 著     电子工业出版社
"""

import cv2
# --------------读取并绘制原始图像------------------
o = cv2.imread('hand.bmp')  
cv2.imshow("original",o)
# --------------提取轮廓------------------
gray = cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
contours, hierarchy = cv2.findContours(binary,
                                             cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_SIMPLE)  
# --------------寻找凸包，得到凸包的角点------------------
hull = cv2.convexHull(contours[0])
# --------------绘制凸包------------------
cv2.polylines(o, [hull], True, (0, 255, 0), 2)
# --------------显示凸包------------------
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()
