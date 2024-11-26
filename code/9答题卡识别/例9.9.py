# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 08:58:17 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）（名称待定）
计算机视觉40个核心案例——从入门到深度学习（python+OpenCV）（名称待定）
李立宗 著     电子工业出版社
"""

import cv2
import numpy as np
thresh=cv2.imread("thresh.bmp",-1)
# cv2.imshow("thresh_original", thresh)
#============查找所有的轮廓======================
cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("共找到各种轮廓",len(cnts),"个")
#============构造载体======================
#thresh：在该图像内显示选项无序时序号
# thresh是灰度图像，变为彩色的是为了能够以彩色显示序号
thresh=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
#============筛选出选项的轮廓======================
options = []     #用于存储筛选出的选项
font = cv2.FONT_HERSHEY_SIMPLEX
for (i,ci) in enumerate(cnts):
    # 获取轮廓的矩形包围框
    x, y, w, h = cv2.boundingRect(ci)
    #ar纵横比
    ar = w / float(h)
    #满足长度、宽度大于25像素，纵横比在[0.6,1.3]之间，加入到options中
    if w >= 25 and h >= 25 and ar >= 0.6 and ar <= 1.3:
        options.append(ci)
        # 绘制序号
        cv2.putText(thresh, str(i), (x-1,y-5), font, 0.5, (0, 0, 255),2)
# 需要注意，此时得到了很多选项的轮廓，但是他们在options是无规则存放的
# print("共找到选项",len(options),"个")
# 绘制每个选项的轮廓
# cv2.drawContours(thresh, options, -1, color, 5)
# ============显示选项无序时图像=====================
cv2.imshow("thresh", thresh)
# =============将轮廓按照从上到下的顺序排序============
boundingBoxes = [cv2.boundingRect(c) for c in options]
(options, boundingBoxes) = zip(*sorted(zip(options, boundingBoxes),
                                    key=lambda x: x[1][1], reverse=False))
# ============将每一题目的四个选项筛选出来===========
for (tn, i) in enumerate(np.arange(0, len(options), 4)):
    # 需要注意，取出的4个轮廓，对应某一道题的4个选项
    # 但是这4个选项的存放是无序的
    # 将轮廓按照坐标实现自左向右顺次存放
    # 将选项A、选项B、选项C、选项D，按照坐标顺次存放
    boundingBoxes = [cv2.boundingRect(c) for c in options[i:i + 4]]
    (cnts, boundingBoxes) = zip(*sorted(zip(options[i:i + 4], boundingBoxes),
                                    key=lambda x: x[1][0], reverse=False))
    # 构造图像image用来显示每道题目的四个选项
    image = np.zeros(thresh.shape, dtype="uint8")
    # 针对每个选项单独处理
    for (n,ni) in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(ni)
        cv2.drawContours(image, [ni], -1, (255, 255, 255), -1)
        cv2.putText(image, str(n), (x-1,y-5), font, 1, (0, 0, 255),2)
    # 显示每个题目的四个选项及对应的序号
    cv2.imshow("result"+str(tn), image)
cv2.waitKey()
cv2.destroyAllWindows()