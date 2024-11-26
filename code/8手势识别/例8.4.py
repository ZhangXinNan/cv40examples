# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 13:26:12 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）(待定名称)
李立宗 著     电子工业出版社
"""
import cv2
# 手势识别函数
def reg(x):
    #=================找出轮廓===============
    #查找所有轮廓
    x=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
    contours,h = cv2.findContours(x,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #从所有轮廓中找到最大的，作为手势的轮廓
    cnt = max(contours,key=lambda x:cv2.contourArea(x))  
    areacnt = cv2.contourArea(cnt)   #获取轮廓面积
    #===========获取轮廓的凸包=============
    hull = cv2.convexHull(cnt)   #获取轮廓的凸包,用于计算面积，返回坐标
    areahull = cv2.contourArea(hull)   #获取凸包的面积
    #===========获取轮廓面积、凸包面积，二者的比值=============
    arearatio = areacnt/areahull  
    #通常情况下，手势0，轮廓和凸包大致相等，该值大于0.9.
    # 手势1，轮廓要比凸包小一些，该值小于等于0.9
    # 需要注意，这个不是特定值，因人而异，有的人手指长，有的人手指短
    # 所以，该值存在一定的差异
    if arearatio>0.9:     #轮廓面积/凸包面积>0.9,二者面积近似，识别为0
            result='fist:0'
    else:
            result='finger:1'  #对应：轮廓面积/凸包面积<=0.9,较大凸缺陷，识别为1
    return result 
# 读取两幅图像识别
x = cv2.imread('zero.jpg')  
y = cv2.imread('one.jpg')  
# 分别识别x和y
xtext=reg(x)
ytext=reg(y)
# 输出识别结果
org=(0,80)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale=2
color=(0,0,255)
thickness=3
cv2.putText(x,xtext,org,font,fontScale,color,thickness)
cv2.putText(y,ytext,org,font,fontScale,color,thickness)
# 显示识别结果
cv2.imshow('zero',x)
cv2.imshow('one',y)
cv2.waitKey()
cv2.destroyAllWindows()

