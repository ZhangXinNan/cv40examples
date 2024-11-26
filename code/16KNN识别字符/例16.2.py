# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 10:43:16 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
import numpy as np
# 导入数据集，并将第1位上字符转换为数字
data= np.loadtxt('letter-recognition.data', dtype= 'float32', delimiter = ',',
                    converters= {0: lambda ch: ord(ch)-ord('A')})
# 将数据集平均划分为训练集合测试集两部分
train, test = np.vsplit(data,2)
# 将训练集、测试集内的标签和特征划分开
responses, trainData = np.hsplit(train,[1])
labels, testData = np.hsplit(test,[1])
# 使用KNN模块
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
ret, result, neighbours, dist = knn.findNearest(testData, k=5)
# 输出结果
correct = np.count_nonzero(result == labels)
accuracy = correct*100.0/10000
print( "识别的准确率为:",accuracy )