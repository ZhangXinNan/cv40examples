# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:15:40 2018
modified on :2021-10-2 18:33

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""
# ====================0．导入库=======================
import numpy as np
import cv2
# ===================1．图像预处理=======================
img = cv2.imread('cat.jpg')
data = img.reshape((-1,3))
data = np.float32(data)
# =================2．使用K均值聚类模块=====================
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(data,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# ====================3．打印的实现============================
print("距离：",ret)
print("标签：\n",label)
print("分类中心点：\n",center)
# ================4．像素值替换及结果展示=======================
center = np.uint8(center)
res1 = center[label.flatten()]
res2 = res1.reshape((img.shape))
cv2.imshow("original",img)
cv2.imshow("result",res2)
cv2.waitKey()
cv2.destroyAllWindows()
