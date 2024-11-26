# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 16:08:51 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
c=cv2.imread("c.bmp")
d=cv2.imread("d.bmp")
# print(c)
e1=c+d
e2=cv2.add(c,d)
e3=cv2.addWeighted(c,1,d,1,0)
cv2.imshow("c",c)
cv2.imshow("d",d)
cv2.imshow("e1",e1)
cv2.imshow("e2",e2)
cv2.imshow("e3",e3)
print("d")
cv2.waitKey()
cv2.destroyAllWindows()
