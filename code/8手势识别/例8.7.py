# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 16:53:48 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）(待定名称)
李立宗 著     电子工业出版社
"""

import cv2

def reg(x):
    o1 = cv2.imread('paper.jpg',1)
    o2 = cv2.imread('rock.jpg',1)
    o3 = cv2.imread('scissors.jpg',1)  
    gray1 = cv2.cvtColor(o1,cv2.COLOR_BGR2GRAY) 
    gray2 = cv2.cvtColor(o2,cv2.COLOR_BGR2GRAY) 
    gray3 = cv2.cvtColor(o3,cv2.COLOR_BGR2GRAY) 
    xgray = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY) 
    ret, binary1 = cv2.threshold(gray1,127,255,cv2.THRESH_BINARY) 
    ret, binary2 = cv2.threshold(gray2,127,255,cv2.THRESH_BINARY) 
    ret, binary3 = cv2.threshold(gray3,127,255,cv2.THRESH_BINARY) 
    xret, xbinary = cv2.threshold(xgray,127,255,cv2.THRESH_BINARY) 
    contours1, hierarchy = cv2.findContours(binary1,
                                                  cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_SIMPLE)  
    contours2, hierarchy = cv2.findContours(binary2,
                                                  cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_SIMPLE)  
    contours3, hierarchy = cv2.findContours(binary3,
                                                  cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_SIMPLE)  
    xcontours, hierarchy = cv2.findContours(xbinary,
                                                  cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_SIMPLE)  
    cnt1 = contours1[0]
    cnt2 = contours2[0]
    cnt3 = contours3[0]
    x = xcontours[0]
    ret=[]
    ret.append(cv2.matchShapes(x,cnt1,1,0.0))
    ret.append(cv2.matchShapes(x,cnt2,1,0.0))
    ret.append(cv2.matchShapes(x,cnt3,1,0.0))
    max_index = ret.index(min(ret))  #计算最大值索引
    if max_index==0:
        r="paper"
    elif max_index==1:
        r="rock"
    else:
        r="sessiors"
    return r

t1=cv2.imread('test1.jpg',1)
t2=cv2.imread('test2.jpg',1)
t3=cv2.imread('test3.jpg',1)
# print(reg(t1))
# print(reg(t2))
# print(reg(t3))
# ===========显示处理结果==================
org=(0,60)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale=2
color=(255,255,255)
thickness=3
cv2.putText(t1,reg(t1),org,font,fontScale,color,thickness)
cv2.putText(t2,reg(t2),org,font,fontScale,color,thickness)
cv2.putText(t3,reg(t3),org,font,fontScale,color,thickness)
cv2.imshow('test1',t1)
cv2.imshow('test2',t2)
cv2.imshow('test3',t3)
cv2.waitKey()
cv2.destroyAllWindows()