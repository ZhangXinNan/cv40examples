# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 17:32:43 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import numpy as np
import cv2
image=np.ones((3,3),np.uint8)*100
print("原始数据：\n",image)
blob = cv2.dnn.blobFromImage(image,scalefactor=1)
print("scalefactor=1：\n",blob[0])
blob = cv2.dnn.blobFromImage(image,scalefactor=0.1)
print("scalefactor=0.1：\n",blob[0])
