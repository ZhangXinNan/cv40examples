# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 13:26:12 2021

@author: Administrator
"""

import cv2
# --------------读取原始图像------------------
zero = cv2.imread('zero.jpg')  
one = cv2.imread('one.jpg')  
# --------------提取zero轮廓，绘制凸包------------------
gray = cv2.cvtColor(zero,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
contours, hierarchy = cv2.findContours(binary,
                                             cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_SIMPLE)  
hull = cv2.convexHull(contours[0])
cv2.polylines(zero, [hull], True, (0, 255, 0), 2)
# --------------提取one轮廓，绘制凸包------------------
gray = cv2.cvtColor(one,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
contours, hierarchy = cv2.findContours(binary,
                                             cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_SIMPLE)  
hull = cv2.convexHull(contours[0])
cv2.polylines(one, [hull], True, (0, 0, 255), 2)

# --------------显示凸包------------------
cv2.imshow("zero-result",zero)
cv2.imshow("one-result",one)
cv2.waitKey()
cv2.destroyAllWindows()
