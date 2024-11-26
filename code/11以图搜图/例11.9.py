# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 08:01:02 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""
import glob
import cv2
import numpy as np
#========提取感知哈希值函数===============
def getHash(I):
    size=(8,8)
    I=cv2.resize(I,size)
    I=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    m=np.mean(I)
    r=(I>m).astype(int)
    x=r.flatten()
    return x
#========计算指定文件夹下的所有图像hash值===============
images = []
EXTS = 'jpg', 'jpeg', 'JPG', 'JPEG', 'gif', 'GIF', 'png', 'PNG','BMP'
for ext in EXTS:
    images.extend(glob.glob('image/*.%s' % ext))
seq = []
for f in images:
    I=cv2.imread(f)
    seq.append((f, getHash(I)))
print(seq)
