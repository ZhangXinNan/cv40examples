# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:04:03 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""


import cv2
img=cv2.imread("lena.bmp",0)
cv2.imshow("before",img)
print("img[50,90]原始值:",img[50,90])
img[10:100,80:100]=255
print("img[50,90]修改值:",img[50,90])
cv2.imshow("after",img)
cv2.waitKey()
cv2.destroyAllWindows()
