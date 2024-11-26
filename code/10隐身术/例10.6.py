# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:30:01 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import numpy as np
import cv2
img1=np.random.randint(0,256,size=[3,3],dtype=np.uint8)
img2=np.random.randint(0,256,size=[3,3],dtype=np.uint8)
print("img1=\n",img1)
print("img2=\n",img2)
img3=cv2.add(img1,img2)
print("cv2.add(img1,img2)=\n",img3)
