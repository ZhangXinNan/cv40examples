# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 08:04:10 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）（名称待定）
计算机视觉40个核心案例——从入门到深度学习（python+OpenCV）（名称待定）
李立宗 著     电子工业出版社
"""

import cv2
thresh=cv2.imread("thresh.bmp",-1)
cv2.imshow("thresh_original", thresh)
#============查找所有的轮廓======================
cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("共找到各种轮廓",len(cnts),"个")
#============筛选出选项的轮廓======================
options = []
for ci in cnts:
    # 获取轮廓的矩形包围框
    x, y, w, h = cv2.boundingRect(ci)
    #ar纵横比
    ar = w / float(h)
    #满足长度、宽度大于25像素，纵横比在[0.6,1.3]之间，加入到options中
    if w >= 25 and h >= 25 and ar >= 0.6 and ar <= 1.3:
        options.append(ci)
# 需要注意，此时得到了很多选项的轮廓，但是他们在options是无规则存放的
print("共找到选项",len(options),"个")
# ===========将所有找到的选项轮廓绘制出来================
color = (0, 0, 255)  #红色
# 为了以彩色显示，将原始图像转换为彩色空间
thresh=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
# 绘制每个选项的轮廓
cv2.drawContours(thresh, options, -1, color, 5)
cv2.imshow("thresh_result", thresh)
cv2.waitKey()
cv2.destroyAllWindows()