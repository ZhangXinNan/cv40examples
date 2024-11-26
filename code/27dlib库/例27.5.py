# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 19:57:51 2021

@author: Administrator
"""

import dlib
import cv2
# 载入模型
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
# 读取图像
img = cv2.imread("people.jpg", cv2.IMREAD_COLOR)
# 检测
faces = cnn_face_detector(img, 1)
# 返回的结果faces是一个mmod_rectangles对象,有2个成员变量：
# dlib.rect类，表示对象的位置
# dlib.confidence，表示置信度。
for i, d in enumerate(faces):
    # 计算每个人脸的位置
    rect = d.rect    
    left = rect.left()
    top = rect.top()
    right = rect.right()
    bottom = rect.bottom()
    # 绘制人脸对应的矩形框
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)    
    cv2.imshow("result", img)
k = cv2.waitKey()
cv2.destroyAllWindows()
