# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 22:09:33 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）（名称待定）
计算机视觉40个核心案例——从入门到深度学习（python+OpenCV）（名称待定）
李立宗 著     电子工业出版社
"""

import cv2
thresh=cv2.imread("thresh.bmp",0)
cv2.imshow("thresh", thresh)
cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("共找到各种轮廓",len(cnts),"个")
threshColor=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
cv2.drawContours(threshColor, cnts, -1, (0,0,255), 3)
cv2.imshow("result",threshColor)
cv2.waitKey()
cv2.destroyAllWindows()
