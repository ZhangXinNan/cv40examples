# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:19:20 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""


import cv2
img=cv2.imread("lenacolor.png")
cv2.imshow("before",img)
print("访问img[0,0]=",img[0,0])
print("访问img[0,0,0]=",img[0,0,0])
print("访问img[0,0,1]=",img[0,0,1])
print("访问img[0,0,2]=",img[0,0,2])
print("访问img[50,0]=",img[50,0])
print("访问img[100,0]=",img[100,0])
#区域1：白色
img[0:50,0:100,0:3]=255
#区域2：灰色
img[50:100,0:100,0:3]=128
#区域3 ：黑色
img[100:150,0:100,0:3]=0 
#区域4 ：红色
img[150:200,0:100]=(0,0,255)          
#显示
cv2.imshow("after",img)
print("修改后img[0,0]=",img[0,0])
print("修改后img[0,0,0]=",img[0,0,0])
print("修改后img[0,0,1]=",img[0,0,1])
print("修改后img[0,0,2]=",img[0,0,2])
print("修改后img[50,0]=",img[50,0])
print("修改后img[100,0]=",img[100,0])
cv2.waitKey()
cv2.destroyAllWindows()

