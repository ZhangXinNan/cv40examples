# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:43:10 2018

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""

import cv2
#读取原始载体图像
lena=cv2.imread("image\lenacolor.png")
#读取水印图像
watermark=cv2.imread("image\watermark.bmp",1)
#按位或运算
e=cv2.bitwise_or(lena,watermark)
#============显示============
cv2.imshow("lena",lena)
cv2.imshow("watermark",watermark)   
cv2.imshow("bitwise_or",e)
#============释放============
cv2.waitKey()
cv2.destroyAllWindows()