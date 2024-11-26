# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 22:07:21 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt 
# ================1．数据准备====================
X = np.random.randint(0,100,(50,2))
X = np.float32(X)
# ==============2．使用K均值聚类模块===============
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(X,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# ==============3．打印的实现=====================
print("距离：",ret)
print("标签：",np.reshape(label,-1))
print("分类中心点：\n",center)
# ==============4．可视化的实现====================
# 根据kmeans处理结果，将数据分类，分为A和B两大类
A = X[label.ravel()==0]
B = X[label.ravel()==1]
# 绘制分类结果数据
plt.scatter(A[:,0],A[:,1],c = 'g', marker = 's')
plt.scatter(B[:,0],B[:,1],c = 'r', marker = 'o')
# 绘制分类数据的中心点
plt.scatter(center[0,0],center[0,1],s = 200,c = 'b', marker = 's')
plt.scatter(center[1,0],center[1,1],s = 200,c = 'b', marker = 'o')
plt.show()