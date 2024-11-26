# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 16:44:33 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""

import cv2
import numpy as np
#读取原始载体图像
lena=cv2.imread("lena.bmp",0)
# 显示原始图像
cv2.imshow("lena",lena)
#读取原始载体图像的shape值
r,c=lena.shape
mask=np.zeros((r,c),dtype=np.uint8)
mask[220:400,250:350]=1
#获取一个key,打码、解码所使用的密钥
key=np.random.randint(0,256,size=[r,c],dtype=np.uint8)
#============获取打码脸============
#step 1:使用密钥key对原始图像lena加密
lenaXorKey=cv2.bitwise_xor(lena,key)
#step 2:获取加密图像的脸部信息encryptFace
encryptFace=cv2.bitwise_and(lenaXorKey,mask*255)
#step3:将图像lena内的脸部值设置为0，得到noFace1
noFace1=cv2.bitwise_and(lena,(1-mask)*255)
#step 4:得到打码的lena图像
maskFace=encryptFace+noFace1
cv2.imshow("maskFace",maskFace)
#============将打码脸解码============
#step 5:将脸部打码的lena与密钥key进行异或运算，得到脸部的原始信息
extractOriginal=cv2.bitwise_xor(maskFace,key)
#step 6:将解码的脸部信息extractOriginal提取出来，得到extractFace
extractFace=cv2.bitwise_and(extractOriginal,mask*255)
#step 7:从脸部打码的lena内提取没有脸部信息的lena图像，得到noFace2
noFace2=cv2.bitwise_and(maskFace,(1-mask)*255)
#step 8:得到解码的lena图像
extractLena=noFace2+extractFace
cv2.imshow("extractLena",extractLena)
cv2.waitKey()
cv2.destroyAllWindows()
