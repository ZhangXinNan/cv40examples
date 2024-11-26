# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 14:34:36 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）(待定名称)
李立宗 著     电子工业出版社
"""

import cv2
#----------------原图--------------------------
img = cv2.imread('hand.bmp')
cv2.imshow('original',img)
#----------------构造轮廓--------------------------
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255,0)
contours, hierarchy = cv2.findContours(binary,
                                             cv2.RETR_TREE,
                                 
            cv2.CHAIN_APPROX_SIMPLE)  
#----------------凸包--------------------------
cnt = contours[0]
hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
print("defects=\n",defects)
#----------------构造凸缺陷--------------------------
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img,start,end,[0,0,255],2)
    cv2.circle(img,far,5,[255,0,0],-1)
#----------------显示结果，释放图像--------------------------
cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
