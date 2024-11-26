# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 22:15:36 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

# ====================导入库======================
import cv2
# ================读取原始图像===================
image = cv2.imread("gua.jpg")       #读取原始图像
rawImage=image.copy()               #复制原始图像
cv2.imshow("original",image)  #测试语句，观察原始图像
# ===========滤波处理O1（去噪）=====================
image = cv2.GaussianBlur(image, (3, 3), 0)
cv2.imshow("GaussianBlur",image)  #测试语句，查看滤波结果（去噪）
# ========== 灰度变换O2（色彩空间转换BGR-->GRAY)===========
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",image)  #测试语句，查看灰度图像
# ==============边缘检测O3（Sobel算子、X方向边缘梯度）===============
SobelX = cv2.Sobel(image, cv2.CV_16S, 1, 0)
absX = cv2.convertScaleAbs(SobelX)  # 映射到[0.255]内
image = absX
cv2.imshow("soblex",image)  #测试语句，图像边缘
# ===============二值化O4（阈值处理）==========================
ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
cv2.imshow("imageThreshold",image)   #测试语句，查看处理结果
# ===========闭运算O5：先膨胀后腐蚀，车牌各个字符是分散的，让车牌构成一体=======
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX)
cv2.imshow("imageCLOSE",image)    #测试语句，查看处理结果
# =============开运算O6：先腐蚀后膨胀，去除噪声==============
kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernelY)
cv2.imshow("imageOPEN",image)
# ================滤波O7：中值滤波，去除噪声=======================
image = cv2.medianBlur(image, 15)
cv2.imshow("imagemedianBlur",image)    #测试语句，查看处理结果
# =================查找轮廓O8==================
contours, w1 = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#测试语句，查看轮廓
image = cv2.drawContours(rawImage.copy(), contours, -1, (0, 0, 255), 3)
cv2.imshow('imagecc', image)
#============定位车牌O9：逐个遍历轮廓，将宽度>3倍高度的轮廓确定为车牌============
for item in contours:
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    if weight > (height * 3):
        plate = rawImage[y:y + height, x:x + weight]   
#================显示提取车牌============================        
cv2.imshow('plate',plate)  # 测试语句：查看提取车牌
cv2.waitKey()
cv2.destroyAllWindows()