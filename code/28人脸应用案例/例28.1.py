# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 20:57:51 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""
from scipy.spatial import distance as dist   #计算欧式距离
import numpy as np
import dlib
import cv2
# 自定义函数，计算嘴的宽高比
# Mouth Aspect Ratio,嘴宽高比。参数也可以直接使用landmarks，确定好关键点位置即可
def MAR(mouth):
    A = dist.euclidean(mouth[3], mouth[9])   #欧氏距离，直接计算y轴差值也可以
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A+B+C)/3
    D = dist.euclidean(mouth[0], mouth[6])
    mar=avg/D
    return mar
# 自定义函数，计算嘴宽度、脸颊宽度的比值
# Mouth Jaw Ratio,嘴宽度/脸颊宽度（嘴/下巴)
def MJR(shape):
    #嘴宽度，欧氏距离，也可以直接计算x轴差值
    mouthWidht = dist.euclidean(shape[48], shape[54])  
    #下巴两侧宽度，根据实际情况选用不同的索引如：4和13等等
    jawWidth = dist.euclidean(shape[3], shape[13])      
    return mouthWidht/jawWidth                          #比值
# 自定义函数，绘制嘴轮廓
def drawMouth(mouth):
    # 针对嘴型构造凸包
    mouthHull = cv2.convexHull(mouth)
    # 把嘴的凸包轮廓绘制出来，便于观察
    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
# 模型初始化
shape_predictor= "shape_predictor_68_face_landmarks.dat" #face_landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
# 初始化摄像头
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
# 逐帧处理
while True:
    # 读取视频放入frame
    _,frame = cap.read()
    # 色彩空间转换彩色(BGR)-->灰度（Gray）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 获取人脸
    rects = detector(gray, 0)
    # 对检测到的rects，逐个遍历
    for rect in rects:
        # 针对脸部的关键点进行处理，构成坐标(x,y)形式
        shape = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
        #获取嘴部的关键点集（在整个脸部索引中，其索引范围为[48,60],不包含61）
        mouth= shape[48:61]
        #计算嘴的宽高比、嘴/脸颊值
        mar = MAR(mouth)   #计算嘴部的高宽比
        mjr = MJR(shape)    #计算“嘴宽/脸颊宽”
        result="normal"   #默认是正常表情
        # 嘴的宽高比，嘴脸颊宽比值，每个人不一样，我用0.5.
        # 大家可以根据实际情况确定不同的值
        # print("mar",mar,"mjr",mjr)  #测试一下实际值，可以根据该值确定
        if mar > 0.5:
            result="laugh"
        elif mjr>0.45 :  # 任意一个超过阈值（都是0.5）为微笑
            result="smile"
        cv2.putText(frame, result, (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #绘制嘴部轮廓
        drawMouth(mouth)   
        # 实时观察mar值        
        # cv2.putText(frame, "MAR: {}".format(mar), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 
        # 实时观察mjr值
        # cv2.putText(frame, "MJR: {}".format(mjr), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)        
    cv2.imshow("Frame", frame)
    # 按下ESC键盘退出（ESC键对应的ASCII码为27）
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()
