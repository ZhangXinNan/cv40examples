# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:43:10 2018

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""
# 处理思路：
# 1.首先将载体图像中与水印图像中文字部分置零
# 2.将水印反色后加载到处理后的载体图像上
import cv2
#读取原始载体图像A
A=cv2.imread("image\lena.bmp",0)
#读取水印图像B
B=cv2.imread("image\watermark.bmp",0)
#将水印B内的255处理为1得到C，也可以使用函数threshold处理
C=B.copy()
w=C[:,:]>0
C[w]=1
#以下均按照图中的关系完成计算
D=A*C
E=255-B
F=D+E
cv2.imshow("A",A)
cv2.imshow("B",B)
cv2.imshow("C",C*255)
cv2.imshow("D",D)
cv2.imshow("E",E)
cv2.imshow("F",F)
cv2.waitKey()
cv2.destroyAllWindows()