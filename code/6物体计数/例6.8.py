# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:39:05 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""
#===================导入库==================
import cv2
#===================读取原始图像==================
img=cv2.imread('count.jpg',1) 
#====================图像预处理===========================
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #色彩空间转换:彩色-->灰度图片
ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) # 阈值处理二值反色
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))#核
erosion=cv2.erode(binary,kernel,iterations=4) #腐蚀操作
dilation=cv2.dilate(erosion,kernel,iterations=3)   #膨胀操作
gaussian = cv2.GaussianBlur(dilation,(3,3),0)# 高斯滤波
#================查找所有轮廓=======================
contours,hirearchy=cv2.findContours(gaussian, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 找出轮廓
#==============筛选出符合要求的轮廓=============
contoursOK=[]   #放置符合要求的轮廓
for i in contours:
     if cv2.contourArea(i)>30:  # 筛选出面积大于30的轮廓 
        contoursOK.append(i)
#==============绘制出符合要求的轮廓=============
draw=cv2.drawContours(img,contoursOK,-1,(0,255,0),1)  #绘制轮廓
#===========计算每一个细胞中心，并绘制数字序号===============
for i,j in zip(contoursOK,range(len(contoursOK))):
    M = cv2.moments(i)
    cX=int(M["m10"]/M["m00"])
    cY=int(M["m01"]/M["m00"])
    cv2.putText(draw, str(j), (cX, cY), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 0, 255), 2) #在中心坐标点上描绘数字
#=============显示图片==================
cv2.imshow("gaussian",gaussian)
cv2.imshow("draw",draw)
#============释放窗口====================
cv2.waitKey()
cv2.destroyAllWindows()