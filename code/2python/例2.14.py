# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:02:07 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
李立宗 著     电子工业出版社
"""


a=input("请输入a:")
b=input("请输入b:")
a=int(a)
b=int(b)
big=(a if a>b else b)
print("大数是:",big)