# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 22:03:35 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）（名称待定）
计算机视觉40个核心案例——从入门到深度学习（python+OpenCV）（名称待定）
李立宗 著     电子工业出版社
"""

import cv2
paper=cv2.imread("paper.jpg",0)
cv2.imshow("paper",paper)
ret,thresh = cv2.threshold(paper, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("thresh", thresh)
cv2.imwrite("thresh.bmp",thresh)
cv2.waitKey()
cv2.destroyAllWindows()