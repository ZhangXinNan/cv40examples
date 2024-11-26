# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:44:22 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）(待定名称)
李立宗 著     电子工业出版社
"""

import cv2
o1 = cv2.imread('o1.jpg')
o2 = cv2.imread('o2.jpg')
o3 = cv2.imread('o3.jpg')  
gray1 = cv2.cvtColor(o1,cv2.COLOR_BGR2GRAY) 
gray2 = cv2.cvtColor(o2,cv2.COLOR_BGR2GRAY) 
gray3 = cv2.cvtColor(o3,cv2.COLOR_BGR2GRAY) 
ret, binary1 = cv2.threshold(gray1,127,255,cv2.THRESH_BINARY) 
ret, binary2 = cv2.threshold(gray2,127,255,cv2.THRESH_BINARY) 
ret, binary3 = cv2.threshold(gray3,127,255,cv2.THRESH_BINARY) 
contours1, hierarchy = cv2.findContours(binary1,
                                              cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)  
contours2, hierarchy = cv2.findContours(binary2,
                                              cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)  
contours3, hierarchy = cv2.findContours(binary3,
                                              cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)  
cnt1 = contours1[0]
cnt2 = contours2[0]
cnt3 = contours3[0]
ret0 = cv2.matchShapes(cnt1,cnt1,1,0.0)
ret1 = cv2.matchShapes(cnt1,cnt2,1,0.0)
ret2 = cv2.matchShapes(cnt1,cnt3,1,0.0)
# ret0 = cv2.matchShapes(gray1,gray1,1,0.0)
# ret1 = cv2.matchShapes(gray1,gray2,1,0.0)
# ret2 = cv2.matchShapes(gray1,gray3,1,0.0)

print("o1.shape=",o1.shape)
print("o2.shape=",o2.shape)
print("o3.shape=",o3.shape)
print("相同图像(cnt1,cnt1)的matchShape=",ret0)
print("相似图像(cnt1,cnt2)的matchShape=",ret1)
print("不相似图像(cnt1,cnt3)的matchShape=",ret2)
cv2.imshow("original1",o1)
cv2.imshow("original2",o2)
cv2.imshow("original3",o3)
cv2.waitKey()
cv2.destroyAllWindows()
