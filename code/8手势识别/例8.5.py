# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 21:59:12 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）(待定名称)
李立宗 著     电子工业出版社
"""

import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    
#==============主程序======================
while(cap.isOpened()):
    ret,frame = cap.read() # 读取摄像头图像
    # print(frame.shape)   #获取窗口大小
    frame = cv2.flip(frame,1)   #沿着y轴转换下方向
    #===============设定一个固定区域作为识别区域=============
    roi = frame[10:210,400:600] # 将右上角设置为固定识别区域
    cv2.rectangle(frame,(400,10),(600,210),(0,0,255),0) # 将选定的区域标记出来
    #===========在hsv色彩空间内检测出皮肤===============
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)    #色彩空间转换
    lower_skin = np.array([0,28,70],dtype=np.uint8)   #设定范围，下限
    upper_skin = np.array([20, 255, 255],dtype=np.uint8)  #设定范围，上限
    mask = cv2.inRange(hsv,lower_skin,upper_skin)   #确定手所在区域
    #===========预处理===============
    kernel = np.ones((2,2),np.uint8)   #构造一个核
    mask = cv2.dilate(mask,kernel,iterations=4)   #膨胀操作
    mask = cv2.GaussianBlur(mask,(5,5),100)       #高斯滤波    
    #=================找出轮廓===============
    #查找所有轮廓
    contours,h = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #从所有轮廓中找到最大的，作为手势的轮廓
    cnt = max(contours,key=lambda x:cv2.contourArea(x))  
    areacnt = cv2.contourArea(cnt)   #获取轮廓面积
    #===========获取轮廓的凸包=============
    hull = cv2.convexHull(cnt)   #获取轮廓的凸包,用于计算面积，返回坐标
    # hull = cv2.convexHull(cnt,returnPoints=False)
    areahull = cv2.contourArea(hull)   #获取凸包的面积
    #===========获取轮廓面积、凸包的面积比=============
    arearatio = areacnt/areahull   
    # 轮廓面积/凸包面积 ：
    # 大于0.9，表示几乎一致，是手势0
    # 否则，说明凸缺陷较大，是手势1.
    #===========获取凸缺陷=============
    hull = cv2.convexHull(cnt,returnPoints=False) #使用索引，returnPoints=False
    defects = cv2.convexityDefects(cnt,hull)    #获取凸缺陷
    #===========凸缺陷处理==================
    n=0 #定义凹凸点个数初始值为0 
    #-------------遍历凸缺陷，判断是否为指间凸缺陷--------------
    for i in range(defects.shape[0]):
        s,e,f,d, = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0]-far[0])**2+(end[1]-far[1])**2)
        #--------计算手指之间的角度----------------
        angle = math.acos((b**2 + c**2 -a**2)/(2*b*c))*57
        #-----------绘制手指间的凸包最远点-------------
        #角度在[20,90]之间的认为是不同手指所构成的凸缺陷
        if angle<=90 and d>20:
            n+=1
            cv2.circle(roi,far,3,[255,0,0],-1)   #用蓝色绘制最远点
        #----------绘制手势的凸包--------------
        cv2.line(roi,start,end,[0,255,0],2) 
    #============通过凸缺陷个数及面积比判断识别结果=================
    if n==0:           #0个凸缺陷，可能为0，也可能为1
        if arearatio>0.9:     #轮廓面积/凸包面积>0.9，判定为拳头，识别为0
            result='0'
        else:
            result='1'   #轮廓面积/凸包面积<=0.9,说明存在很大的凸缺陷，识别为1
    elif n==1:        #1个凸缺陷，对应2根手指，识别为2
        result='2'
    elif n==2:        #2个凸缺陷，对应3根手指，识别为3
        result='3'
    elif n==3:        #3个凸缺陷，对应4根手指，识别为4
        result='4'
    elif n==4:        #4个凸缺陷，对应5根手指，识别为5
        result='5'
    #============设置与显示识别结果相关的参数=================
    org=(400,80)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale=2
    color=(0,0,255)
    thickness=3
    #================显示识别结果===========================
    cv2.putText(frame,result,org,font,fontScale,color,thickness)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(25)& 0xff  
    if k == 27:     # 键盘Esc键退出
        break
cv2.destroyAllWindows()
cap.release()