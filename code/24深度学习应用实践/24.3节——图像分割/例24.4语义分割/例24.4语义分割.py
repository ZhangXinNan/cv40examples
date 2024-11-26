# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 08:15:43 2021

@author: Administrator
"""
# model:
# http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel
# config:
# https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/fcn8s-heavy-pascal.prototxt
# classes:
# https://github.com/opencv/opencv/blob/master/samples/data/dnn/object_detection_classes_pascal_voc.txt
# 注意：在下载的object_detection_classes_pascal_voc.txt的第一行添加background，不然会报错

import cv2
import numpy as np
# =================读取原始图像=================
image = cv2.imread("a.jpg")
H,W = image.shape[:2]   #获取图像的尺寸：宽和高
# =================导入分类文件=================
classes =  open('object_detection_classes_pascal_voc.txt', 'rt').read().strip().split("\n")
# ================绘制色卡（颜色标识）=====================
# 设置一组随机色，让每个类别使用不同的颜色标识
classesCOLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8") 
classesCOLORS[0] =(0,0,0)  #把背景设置为黑色
# 色卡就是把每个类别的颜色在表格的一行内展示出来（让每个颜色有一定高度，方便观察）
rowHeight = 30  #每种颜色的高度
colorChart = np.zeros((rowHeight * len(classesCOLORS), 200, 3), np.uint8)  #初始化色卡
for i in range(len(classes)):  #根据COLORS配置色卡的文字说明、颜色演示
    # row，色卡的一个颜色条所在行（有高度的，rowHeight）
    row = colorChart[i * rowHeight:(i + 1) * rowHeight]
    # 设置当前遍历到的颜色条的颜色
    row[:,:] = classesCOLORS[i]
    # 设置当前遍历到的颜色条的文字说明
    cv2.putText(row, classes[i], (0, rowHeight//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
cv2.imshow('colorChart', colorChart)
# =================模型推理过程=================
model="fcn8s-heavy-pascal.caffemodel"
config="fcn8s-heavy-pascal.prototxt"
net = cv2.dnn.readNet(model, config)
blob = cv2.dnn.blobFromImage(image, 1.0, (W,H), (0, 0, 0), False, crop=False)
net.setInput(blob)
score = net.forward()
#=============根据推理结果，将每个像素用其所属类颜色构建一个新的模板mask=================
classIDS = np.argmax(score[0], axis=0)   #获取每个像素点所属的分类ID
print(classIDS.shape)
# 根据classIDS确定模板mask
# 每个像素点的颜色为色卡所指定的颜色（色卡颜色来源于classesCOLORS）
mask = np.stack([classesCOLORS[index] for index in classIDS.flatten()])
mask = mask.reshape(H, W, 3)   #调整模板mask为图像的尺寸大小
result =cv2.addWeighted(image,0.2,mask,0.8,0)   #将图像image和模板mask进行加权和计算
cv2.imshow("result",result)
cv2.waitKey(0)
cv2.destroyAllWindows()