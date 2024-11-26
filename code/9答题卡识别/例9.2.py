# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 07:53:41 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）（名称待定）
计算机视觉40个核心案例——从入门到深度学习（python+OpenCV）（名称待定）
李立宗 著     电子工业出版社
"""
import cv2
img = cv2.imread('b.jpg')
cv2.imshow("orginal",img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("gaussian",gaussian)
edged=cv2.Canny(gaussian,50,200) 
cv2.imshow("edged",edged)
cts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, cts, -1, (0,0,255), 3)
cv2.imshow("img",img)
cv2.waitKey()
cv2.destroyAllWindows()
