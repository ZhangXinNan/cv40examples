# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:37:22 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
import dlib
# 摄像头初始化
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
# dlib初始化
detector=dlib.get_frontal_face_detector()
# 针对每一帧进行处理
while True:
    # 捕获一帧
    ret,img=cap.read()
    # 没有捕获到帧，直接退出
    if ret is None:
        break
    # 可以将当前帧处理为灰度，方便后续计算
    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #  参数gray：当前的视频帧（或者图像），参数1：上采样人脸（将人脸放大）
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
    # 显示当前帧，及捕获到的各个人脸框
    cv2.imshow("face",img)
    # 捕获按键
    key=cv2.waitKey(1)
    # 如果按下Ecs键，则退出（Esc的ASCII码为27）
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()
        