# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 20:53:16 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import numpy as np
import dlib
import cv2
# 模型初始化
shape_predictor= "shape_predictor_68_face_landmarks.dat" #dace_landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
# 自定义函数drawLine，将指定的点连接起来
def drawLine(start,end):
    # 获取点集
    pts = shape[start:end]
    # 遍历点集，将各个点用直线连接起来
    for l in range(1, len(pts)):
        ptA = tuple(pts[l - 1])
        ptB = tuple(pts[l])
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
# 自定义函数，将指定的点构成一个凸包、绘制其轮廓
def drawConvexHull(start,end):
    # 注意，凸包用来绘制眼睛、嘴
    # 眼睛、嘴也可以用drawLine通过直线绘制
    # 但是，使用凸包绘制轮廓，更方便进行颜色填充等设置
    # 获取某个特定五官的点集
    Facial = shape[start:end]
    # 针对该五官构造凸包
    mouthHull = cv2.convexHull(Facial)
    # 把凸包轮廓绘制出来
    cv2.drawContours(image, [mouthHull], -1, (0, 255, 0), 2)
# 读取图像
image=cv2.imread("image.jpg")
# 色彩空间转换彩色(BGR)-->灰度（Gray）
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 获取人脸
faces = detector(gray, 0)
# 对检测到的rects，逐个遍历
for face in faces:
    # 针对脸部的关键点进行处理，构成坐标(x,y)形式
    shape = np.matrix([[p.x, p.y] for p in predictor(gray, face).parts()])
    # ============使用函数drawConexHull绘制嘴、眼睛=========================
    #获取嘴部的关键点集（在整个脸部索引中，其索引范围为[48,60],不包含61）
    drawConvexHull(48,59)
    # 嘴内部
    drawConvexHull(60,68)
    # 左眼
    drawConvexHull(42,48)
    # 右眼
    drawConvexHull(36,42)
    # ============使用函数drawLine绘制脸颊、眉毛、鼻子=========================
    # 将shape转换为np.array
    shape=np.array(shape)
    # 绘制脸颊，把脸颊的各个关键点（索引0-16，不含17）用线条连接起来
    drawLine(0,17)
    # 绘制左眉毛，通过将关键点连接实现（索引18-21）
    drawLine(17,22)
    # 绘制右眉毛（索引23-26）
    drawLine(22,27)
    # 鼻子（索引27-36）
    drawLine(27,36)
cv2.imshow("Frame", image)
cv2.waitKey()
cv2.destroyAllWindows()

