# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 08:52:07 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
#========提取感知哈希值函数===============
def getHash(I):
    size=(8,8)
    I=cv2.resize(I,size)
    I=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    m=np.mean(I)
    r=(I>m).astype(int)
    x=r.flatten()
    return x
#==========构造计算汉明距离函数============
def hamming(h1, h2):
    r=cv2.bitwise_xor(h1,h2)
    h=np.sum(r)
    return h
#========计算检索图像的hash值===============
o=cv2.imread("apple.jpg")
h=getHash(o)
print("检索图像的感知哈希值为：\n",h)
#========计算指定文件夹下的所有图像hash值===============
images = []
EXTS = 'jpg', 'jpeg', 'gif', 'png', 'bmp'
for ext in EXTS:
    images.extend(glob.glob('image/*.%s' % ext))
seq = []
for f in images:
    I=cv2.imread(f)
    seq.append((f, getHash(I)))
# print(seq)
#========以图搜图核心：找出最相似图像===============
# 计算检索图像与图像库内所有图像距离，将最小距离作为检索结果
distance=[]
for x in seq:
    distance.append((hamming(h,x[1]),x[0]))   #每次添加（距离值，图像名称）
# print(distance)   #测试代码：看看距离值都是多少
s=sorted(distance)   #排序，把距离最小的放在最前面
# print(s)        #测试代码：看看图像库内各个图像的距离值
r1=cv2.imread(str(s[0][1]))
r2=cv2.imread(str(s[1][1]))
r3=cv2.imread(str(s[2][1]))
# ================绘制结果===================
plt.figure("result")
plt.subplot(141),plt.imshow(cv2.cvtColor(o,cv2.COLOR_BGR2RGB)),plt.axis("off")
plt.subplot(142),plt.imshow(cv2.cvtColor(r1,cv2.COLOR_BGR2RGB)),plt.axis("off")
plt.subplot(143),plt.imshow(cv2.cvtColor(r2,cv2.COLOR_BGR2RGB)),plt.axis("off")
plt.subplot(144),plt.imshow(cv2.cvtColor(r3,cv2.COLOR_BGR2RGB)),plt.axis("off")
plt.show()









