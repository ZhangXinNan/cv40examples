# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:20:01 2021

@author: Administrator
"""
import cv2
#========读取图像=============
image = cv2.imread('tute.jpg')
# 获取图像尺寸
(H, W) = image.shape[:2]
#========加载模型、推理=============
# 加载模型
net = cv2.dnn.readNetFromTorch('model\eccv16\starry_night.t7')
# net = cv2.dnn.readNetFromTorch('model\instance_norm\mosaic.t7')
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (0, 0, 0), swapRB=False, crop=False)
# 推理
net.setInput(blob)
out = net.forward()
print(out)
# out是四维的：B*C*H*W
# B,batch图像数量（通常为1），C：channels通道数，H：height高度、W：width宽度
# ======输出处理=========
# 重塑形状(忽略第1维)，4维变3维
# 调整输出out的形状,模型推理输出out是四维BCHW形式的，调整为三维CHW形式
out = out.reshape(out.shape[1], out.shape[2], out.shape[3])
# 将输出进行归一化处理
cv2.normalize(out, out,alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
# out /= 255 # 修正后，数学运算也可以
# （通道,高度,宽度）转化为（高度,宽度,通道）
result = out.transpose(1, 2, 0)
# ======输出图片=========
cv2.imshow('original', image)
cv2.imshow('result', result)
cv2.waitKey()
cv2.destroyAllWindows()