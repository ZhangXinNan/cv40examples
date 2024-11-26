# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:35:02 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""


import cv2
lena=cv2.imread("lenacolor.png")
cv2.imshow("lena",lena)
b=lena[:,:,0]
g=lena[:,:,1]
r=lena[:,:,2]
cv2.imshow("b",b)
cv2.imshow("g",g)
cv2.imshow("r",r)
lena[:,:,0]=0
cv2.imshow("lenab0",lena)
lena[:,:,1]=0
cv2.imshow("lenab0g0",lena)
cv2.waitKey()
cv2.destroyAllWindows()
