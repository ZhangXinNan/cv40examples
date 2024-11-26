# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:10:17 2020

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""
import cv2
import numpy as np
# 初始化
cap = cv2.VideoCapture(0)
# 获取背景信息
ret,back = cap.read()
# 实时采集
while(cap.isOpened()):
    # 实时采集摄像头信息
	ret, fore = cap.read()
    # 没有捕捉到任何信息，中断
	if not ret:
		break
    # 实时显示采集到的摄像头视频信息
	cv2.imshow('fore',fore)
    # 色彩空间转换BGR-->HSV
	hsv = cv2.cvtColor(fore, cv2.COLOR_BGR2HSV)
    # 红色区间1
	redLower = np.array([0,100,100])
	redUpper = np.array([10,255,255])
    # 红色在hsv空间内的范围1
	maska = cv2.inRange(hsv,redLower,redUpper)
    # 	cv2.imshow('mask1',mask1)    
    # 红色区间2
	redLower = np.array([160,100,100])
	redUpper = np.array([179,255,255])
    # 红色在hsv空间内的范围2
	maskb = cv2.inRange(hsv,redLower,redUpper)
    # 	cv2.imshow('mask2',mask2)    
    # 红色整体区间 = 红色区间1+红色区间2
	mask1 = maska+maskb
    # 	cv2.imshow('mask12',mask1) 
    # 膨胀
	mask1 = cv2.dilate(mask1,np.ones((3,3),np.uint8),iterations = 1)
    # 	cv2.imshow('maskdilate',mask1) 
    # 按位取反 ,黑变白、白变黑   
	mask2 = cv2.bitwise_not(mask1)
    # 	cv2.imshow('maskNot',mask2)
    # 提取back中，mask1指定范围
	result1 = cv2.bitwise_and(back,back,mask=mask1)
    # 	cv2.imshow('res1',res1)
    # 提取fore中，mask2指定的范围
	result2 = cv2.bitwise_and(fore,fore,mask=mask2)
    # 	cv2.imshow('res2',res2)
    # 将res1和res2相加
	result =  result1 + result2
    # 显示最终结果  
	cv2.imshow('result',result)
	k = cv2.waitKey(10)
	if k == 27:
		break
cv2.destroyAllWindows()