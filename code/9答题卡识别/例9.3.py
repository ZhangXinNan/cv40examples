# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:06:47 2018
@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）（名称待定）
计算机视觉40个核心案例——从入门到深度学习（python+OpenCV）（名称待定）
李立宗 著     电子工业出版社
"""
#  测试1，如果是梯形，输出的轮廓点个数是许多个(本例题图片418个点)
#  测试2，如果是矩形，输出的轮廓点个数是4个
# 计算逼近多边形后：
#  测试1，如果是梯形，输出的轮廓点个数是4个
#  测试2，如果是矩形，输出的轮廓点个数是4个
#输出边缘和结构信息
import cv2
o1 = cv2.imread('xtest.jpg')  
cv2.imshow("original1",o1)
o2 = cv2.imread('xtest2.jpg')  
cv2.imshow("original2",o2)
cv2.waitKey()
cv2.destroyAllWindows()
def cstNum(x):
    gray = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)  
    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
    csts, hierarchy = cv2.findContours(binary,
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)  
    print("轮廓具有的顶点的个数：",len(csts[0]))
    peri=0.01*cv2.arcLength(csts[0],True)
    # 获取多边形的所有定点，如果是四个定点，就代表是矩形
    approx=cv2.approxPolyDP(csts[0],peri,True)
    # 打印定点个数
    print("逼近多边形的顶点个数：",len(approx))
print("首先，观察一下梯形：")
cstNum(o1)
print("接下来，观察一下矩形：")
cstNum(o2)