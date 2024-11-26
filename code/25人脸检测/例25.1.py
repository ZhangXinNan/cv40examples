# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 18:22:03 2021

@author: 李立宗  lilizong@gmail.com
《opencv图穷匕见-python实现》 电子工业出版社
"""

import cv2
# ===============1 原始图像处理====================
image = cv2.imread('manyPeople.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# ================2 加载分类器========================
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# =================3 人脸检测========================
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor = 1.04,
    minNeighbors = 18,
    minSize = (8,8))
# ===============4 打印输出的实现=====================
print("发现{0}张人脸!".format(len(faces)))
print("其位置分别是：")
print(faces)
# ==================5 标注人脸及显示=======================
for(x,y,w,h) in faces:
  cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2) 
cv2.imshow("result",image)
cv2.waitKey(0)
cv2.destroyAllWindows()