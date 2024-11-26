# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 19:56:39 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

# ================导入库===================
import cv2
import numpy as np
# ==============1. 生成模拟数据及标签=====================
trainData = np.array([[5,6] ,[9,8],[3,8],[99,94],[89,91],[92,96]]).astype(np.float32)
tdLable = np.array([[0],[0],[0],[1],[1],[1]]).astype(np.float32)
test=np.array([[31,28]]).astype(np.float32)
#===============2. 使用KNN算法=====================
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, tdLable)
ret, results, neighbours, dist = knn.findNearest(test,3)
#===============3. 显示结果=====================
print("当前数可以判定为类型：", results[0][0].astype(int))
print("距离当前点最近的3个邻居是：", neighbours)
print("3个最近邻居的距离: ", dist)

