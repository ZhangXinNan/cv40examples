# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:43:10 2018

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社

"""

import cv2
import numpy as np
#读取原始载体图像
lena=cv2.imread("image\lena.bmp",0)
#============嵌入过程============
#step 1：生成内部值都是254的数组
r,c=lena.shape  #读取原始载体图像的shape值
t1=np.ones((r,c),dtype=np.uint8)*254
#step 2：获取lena图像的高7位,最低有效位置零
lsb0=cv2.bitwise_and(lena,t1)
#step 3：水印信息处理
w=cv2.imread("image\watermark.bmp",0)  # 读取水印图像
#将水印内的255处理为1，以方便嵌入
#也可以使用threshold处理。
wt=w.copy()
wt[w>0]=1
#step 4：将watermark嵌入到lenaH7内
wo=cv2.bitwise_or(lsb0,wt)
#============提取过程============
#step 5：生成内部值都是1的数组
t2=np.ones((r,c),dtype=np.uint8)
#step 6：从载体图像内，提取水印图像
ewb=cv2.bitwise_and(wo,t2)
#step 7：将水印内的1处理为255以方便显示
#也可以使用threshold实现。
ew=ewb
ew[ewb>0]=255
#============显示============
cv2.imshow("lena",lena)  #原始图像
cv2.imshow("watermark",w)   #原始水印图像
cv2.imshow("wo",wo)  #含水印载体
cv2.imshow("ew",ew)  #提取得到的水印
cv2.waitKey()
cv2.destroyAllWindows()