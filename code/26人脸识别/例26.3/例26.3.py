# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 21:33:03 2018

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
import numpy as np
# 读取训练图像
images=[]
images.append(cv2.imread("f01.png",cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("f02.png",cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("f11.png",cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("f12.png",cv2.IMREAD_GRAYSCALE))
# 给训练图像贴标签
labels=[0,0,1,1]
# 读取待识别图像
predict_image=cv2.imread("fTest.png",cv2.IMREAD_GRAYSCALE)
# 识别
recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.train(images, np.array(labels)) 
label,confidence= recognizer.predict(predict_image) 
# 打印识别结果
print("识别标签label=",label)
print("置信度confidence=",confidence)

