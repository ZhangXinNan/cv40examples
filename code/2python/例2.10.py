# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:51:39 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""


a = {
     "李立宗": 66,
     "刘能": 88,
     "赵四": 99
     }
print(a)
#修改李立宗的成绩
a["李立宗"]=90
print(a)
#增加小明及成绩
a['小明']=100
print(a)
#删除李立宗的成绩
del a['李立宗']
print(a)

