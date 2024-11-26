# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 15:16:13 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
import matplotlib.pyplot as plt
# 读取图像
o1=cv2.imread("image/fruit.jpg")
o2=cv2.imread("image/sunset.jpg")
o3=cv2.imread("image/tomato.jpg")
# 绘制结果
plt.figure("result")
plt.subplot(131),plt.imshow(cv2.cvtColor(o1,cv2.COLOR_BGR2RGB)),plt.axis("off")
plt.subplot(132),plt.imshow(cv2.cvtColor(o2,cv2.COLOR_BGR2RGB)),plt.axis("off")
plt.subplot(133),plt.imshow(cv2.cvtColor(o3,cv2.COLOR_BGR2RGB)),plt.axis("off")
plt.show()