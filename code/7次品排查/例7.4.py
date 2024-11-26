# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:54:33 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""
# ============步骤0：导入使用的库=================
import cv2
import numpy as np
# ============步骤1：读取原始图像=================
img = cv2.imread('pill3.jpg')
cv2.imshow("original",img)
# ============步骤2：预处理=================
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('thresh',thresh)
kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))#核
opening1 = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
cv2.imshow('opening1',opening1)
# ============步骤3：使用函数distanceTransform()确定前景=================
dist_transform = cv2.distanceTransform(opening1,cv2.DIST_L2,3)
ret, fore = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
cv2.imshow('fore',fore)
# ============步骤4：去噪处理=================
kernel = np.ones((3,3),np.uint8)
opening2 = cv2.morphologyEx(fore, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening2',opening2)
# ============步骤5：提取轮廓=================
opening2 = np.array(opening2,np.uint8)
contours, hierarchy = cv2.findContours(opening2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# ============步骤6：缺陷检测=================
count=0
font=cv2.FONT_HERSHEY_COMPLEX
for cnt in contours:
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    circle_img = cv2.circle(opening2,center,radius,(255,255,255),1)
    area = cv2.contourArea(cnt)
    area_circle=3.14*radius*radius
    if area/area_circle >=0.5:
        img=cv2.putText(img,'OK',center,font,2,(255,255,255),3)
    else:
        img=cv2.putText(img,'BAD',center,font,2,(255,255,255),3)
    count+=1
img=cv2.putText(img,('sum='+str(count)),(20,50),font,2,(255,255,255))
# ============步骤7：显示处理结果=================
cv2.imshow('result',img)
cv2.waitKey()
cv2.destroyAllWindows()