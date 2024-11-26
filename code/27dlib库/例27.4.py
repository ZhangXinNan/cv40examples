# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:26:21 2021

@author: Administrator
"""


import cv2
import dlib
import numpy as np
# 读入图片
img = cv2.imread("rotate.jpg")
# 步骤1：初始化
# 构造检测器
detector = dlib.get_frontal_face_detector()
# 检测人脸框
faceBoxs = detector(img, 1)
# 载入模型
predictor  = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# 步骤2：获取人脸框集合
# 将步骤1中所获取的人脸框集合faceBoxs中的每个人脸框，逐个放入容器faces中。
faces = dlib.full_object_detections()   #构造容器
for faceBox in faceBoxs:
    faces.append(predictor(img, faceBox))  #把每个人的人脸框放入容器faces内
# 步骤3：根据原始图像、人脸关键点获取人脸对齐结果
# 调用函数get_face_chips完成对人脸图像的对齐（倾斜校正）。
faces = dlib.get_face_chips(img, faces, size=120)
# 步骤4：将获取的每一个人脸显示出来
# 通过循环，将faces中的每一张人脸进行可视化展示。
n = 0       # 用变量n给识别的人脸按顺序编号
# 显示每一个人脸
for face in faces:
    n+=1
    face = np.array(face).astype(np.uint8)
    cv2.imshow('face%s'%(n), face)
# 显示
cv2.imshow("original",img)
cv2.waitKey(0)
cv2.destroyAllWindows()