# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:53:19 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""

import cv2
o = cv2.imread('opencv.png')  
cv2.imshow("original",o)
gray = cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
contours, hierarchy = cv2.findContours(binary,
                                             cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_SIMPLE)  
area=[] 
contoursOK=[]  
for i in contours:
     if cv2.contourArea(i)>1000: 
        contoursOK.append(i)
cv2.drawContours(o,contoursOK,-1,(0,0,255),8) 
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()
