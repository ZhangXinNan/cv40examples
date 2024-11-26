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
lena=cv2.imread("image\lena.bmp",0)
#读取水印图像
watermark=cv2.imread("image\watermark.bmp",0)
#水印图像取反
rWatermark=255-watermark
#加法运算符“+”运算
add1=lena+rWatermark
#add加法运算
add2=cv2.add(lena,rWatermark)
#加权和cv2.addWeighted
add3=cv2.addWeighted(lena,0.6,rWatermark,0.3,55)
#显示
cv2.imshow("lena",lena)
cv2.imshow("watermark",watermark)
cv2.imshow("rWatermark",rWatermark)
cv2.imshow("add1",add1)
cv2.imshow("add2",add2)  
cv2.imshow("add3",add3)  
cv2.waitKey()
cv2.destroyAllWindows()