# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 17:59:57 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""

import cv2
import numpy as np
#读取原始载体图像
lena=cv2.imread("lena.bmp",0)
cv2.imshow("lena",lena)
#读取原始载体图像的shape值
r,c=lena.shape
#设置ROI区域
roi=lena[220:400,250:350]
#获取一个key,打码、解码所使用的密钥
key=np.random.randint(0,256,size=[r,c],dtype=np.uint8)
#============脸部打码过程============
#step 1：使用密钥key加密原始图像lena（按位异或）
lenaXorKey=cv2.bitwise_xor(lena,key) 
#step 2：获取加密后图像的脸部区域（获取ROI）
secretFace=lenaXorKey[220:400,250:350]
cv2.imshow("secretFace",secretFace)
#step 3：划定ROI，其实没有实质性操作
#lena[220:400,250:350] 
#step 4：将lena的脸部区域，替换为加密后的脸部区域secretFace（ROI替换）
lena[220:400,250:350]=secretFace
enFace=lena  #lena已经是处理结果，为了方便理解使用enFace重新命名
cv2.imshow("enFace",enFace)
#============脸部解码过程============
#step 5:将脸部打码的lena与密钥key异或，得到脸部的原始信息（按位异或）
extractOriginal=cv2.bitwise_xor(enFace,key)
#step 6:获取解密后图像的脸部区域（获取ROI）
face=extractOriginal[220:400,250:350]
cv2.imshow("face",face)
#step 7：划定ROI，其实没有实质性操作。
#enFace[220:400,250:350]
#step 8:将enFace的脸部区域，替换为解密的脸部区域face（ROI替换）
enFace[220:400,250:350]=face
deFace=enFace #enFace已经是处理结果，为了方便理解使用deFace重新命名
cv2.imshow("deFace",deFace)
cv2.waitKey()
cv2.destroyAllWindows()
