# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 07:49:34 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""
import numpy as np
import cv2 
# 【step1：预处理】读入文件、色彩空间转换
img = cv2.imread('digits.png')
# 灰度转换：BGR模式-->灰度图像
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 【step2：拆分为独立数字】
# 将原始图像划分成独立的数字，每个数字大小20*20，共计5000个
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
# 装进array，形状(50,100,20,20),50行，100列，每个图像20*20大小
x = np.array(cells)
# 【step3：拆分为训练集和测试集】
# 划分为训练集和测试集：比例各占一半
train = x[:,:50]
test = x[:,50:100]
# 【step4：塑形为符合KNN的输入】
# 数据调整，将每个数字的尺寸由20*20调整为1*400（一行400个像素）
train =train.reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = test.reshape(-1,400).astype(np.float32) # Size = (2500,400)
print(train.shape)
# 【step5：分配标签】
# 分别为训练数据、测试数据分配标签（图像对应的实际值）
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = np.repeat(k,250)[:,np.newaxis]
# 【step6：KNN工作】
# 核心代码：初始化、训练、预测
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)
# 【step7：验证结果】
# 通过测试集校验准确率
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print( "当前使用KNN识别手写数字的准确率为:",accuracy )