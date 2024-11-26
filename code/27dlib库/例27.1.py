# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:37:22 2021

@author: Administrator
"""

import cv2
import dlib
# dlib初始化
detector=dlib.get_frontal_face_detector()
# 读取原始图像
img=cv2.imread("people.jpg")
# 使用人脸检测器返回检测到的人脸框
faces=detector(img,1)
# 针对捕获到的多个人脸进行逐个处理
for face in faces:
    # 获取人脸框的坐标
    x1=face.left()
    y1=face.top()
    x2=face.right()
    y2=face.bottom()
    # 绘制人脸框
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
# 显示捕获到的各个人脸框
cv2.imshow("result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
        