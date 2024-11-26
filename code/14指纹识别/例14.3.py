# -*- coding: utf-8 -*-
"""


@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""
import numpy as np
import cv2
#==========读取、显示指纹图像============
fp= cv2.imread("fingerprint.png")
cv2.imshow("fingerprint",fp)
#==========SIFT==================
sift = cv2.SIFT_create()  #SIFT专利已经到期，可以正常使用。要安装贡献包：opencv-contrib-python
kp, des = sift.detectAndCompute(fp, None)
#==========绘制关键点==================
cv2.drawKeypoints(fp,kp,fp)
#==========显示关键点信息、描述符==================
print("关键点个数：",len(kp))              #显示kp的长度
print("前五个关键点：",kp[:5])             #显示前5条数据
print("第一个关键点的坐标：",kp[0].pt)
print("第一个关键点的区域：",kp[0].size)
print("第一个关键点的角度：",kp[0].angle)
print("第一个关键点的响应：",kp[0].response)
print("第一个关键点的层数：",kp[0].octave)
print("第一个关键点的类id：",kp[0].class_id)
print("描述符形状:",np.shape(des))         #显示des的形状
print("第一个描述符:",des[0])              #显示des[0]的值
#==========可视化关键点==================
cv2.imshow("points",fp)
cv2.waitKey()
cv2.destroyAllWindows()
