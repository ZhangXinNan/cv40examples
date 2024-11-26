# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 21:54:31 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

# ================导入库===================
import cv2
import numpy as np
# ==============1. 生成模拟数据及标签=====================
data = np.array([[6,3] ,[4,5],[9,8],[12,12],[15,13],[18,17]]).astype(np.float32)
label = np.array([[0],[0],[0],[1],[1],[1]]).astype(np.int32)
test=np.array([[12,18]]).astype(np.float32)
# ===============2. SVM分类器=====================
svm = cv2.ml.SVM_create() 
svm.train(data,cv2.ml.ROW_SAMPLE,label)
(p1,p2) = svm.predict(test)
# ============3. 显示分类结果分类==================
rv = p2[0][0].astype(np.int32)
if rv==0  :
    print("当前钻石等级：乙级")
else:
    print("当前钻石等级：甲级")
