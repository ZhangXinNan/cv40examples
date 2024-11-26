# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:43:48 2021

@author: Administrator
"""

# classes:
# https://github.com/opencv/opencv/blob/master/samples/data/dnn/object_detection_classes_pascal_voc.txt
# 注意：在下载的object_detection_classes_pascal_voc.txt的第一行添加background，不然会报错


import numpy as np
import cv2
# ==============读取待检测图像=======================
image = cv2.imread("test2.jpg")
(H, W) = image.shape[:2]  #获取高度和宽度
# ==============读取类别文件=======================
# 导入、处理分类文件
classes =  open('object_detection_classes_pascal_voc.txt', 'rt').read().strip().split("\n")
# 类别文件内存储的是：background、aeroplane、bicycle、bird、boat等分类名称
# 为每个分类随机分配一个颜色
classesCOLORS = (np.random.uniform(0, 255, size=(len(classes), 3)))
# ==============模型导入、推理=======================
config="MobileNetSSD_deploy.prototxt.txt"
model="MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(config, model)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),0.007843, (300, 300), 127.5)
net.setInput(blob)
outs = net.forward()
print(outs.shape)
# outs.shape=[1,1,候选框个数,7]
# outs[1,1,:,1]-->当前候选框对应类别在classes内的索引值
# outs[1,1,:,2]-->当前候选框所对应类别的置信度
# outs[1,1,:,3:7]-->当前候选框的位置信息（左上角、右下角坐标值）
# ==============绘制目标检测结果=======================
# 显示每个置信度大于0.5的对象
# outs.shape[2]-->候选框个数
for i in np.arange(0, outs.shape[2]):  # 逐个遍历各个候选框
    # 获取置信度，用于：1判断当前对象是否显示、2显示用
    confidence = outs[0, 0, i, 2]
    if confidence >0.3 :  #将置信度大于0.3的对象显示出来，忽略置信度小的对象
        # 获取当前候选框对应类别在classes内的索引值
        index = int(outs[0, 0, i, 1])
        # 获取当前候选框的位置信息（左上角、右下角坐标值）
        box = outs[0, 0, i, 3:7] * np.array([W, H, W, H])
        # 获取左上角、右下角坐标值
        (x1,y1,x2,y2) = box.astype("int")
        # 类别标签及置信度
        result = "{}: {:.0f}%".format(classes[index],confidence * 100)
        # 绘制边框
        cv2.rectangle(image, (x1,y1), (x2,y2),classesCOLORS[index], 2)
        # 绘制类别标签
        cv2.putText(image, result, (x1, y1+25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, classesCOLORS[index], 2)
cv2.imshow("result", image)
cv2.waitKey()
cv2.destroyAllWindows()
