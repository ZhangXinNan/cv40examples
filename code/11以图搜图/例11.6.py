# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 21:46:32 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import numpy as np
# 测试ravel，结果是视图，影响原始值
a=np.array([1,2,3,4])  #为了直观，用的一维数组
ar=a.ravel()
ar[1]=666
print("a=",a)
print("ar=",ar)
# 测试flatten，结果是复制品（拷贝，copy），不影响原始值
b=np.array([1,2,3,4])
bf=a.flatten()
bf[1]=666
print("b=",b)
print("bf=",bf)
