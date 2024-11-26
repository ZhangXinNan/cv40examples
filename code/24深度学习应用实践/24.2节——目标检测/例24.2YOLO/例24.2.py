# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 09:28:52 2021

@author: Administrator
"""

import cv2
import numpy as np
# ===========初始化、推理===========
image = cv2.imread("test2.jpg")
# ===========初始化、推理===========
classes =  open('coco.names', 'rt').read().strip().split("\n")
# classes内包含80个不同的类别对象
# 例如，其中的部分类别为：person、bicycle、car、motorbike、aeroplane
# ===========初始化、推理===========
# 步骤1：读取网络模型
net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
# 步骤2：图像预处理
blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (416, 416), (0, 0, 0), True, crop=False)
# 步骤3：设置网络
net.setInput(blob)
# 步骤4：运算
outInfo = net.getUnconnectedOutLayersNames() 
outs = net.forward(outInfo)
# 返回值：
# outs包含3层。
# 第0层：存储着找到的所有可能的较大尺寸的对象
# 第1层：存储着找到的所有可能的中等尺寸的对象
# 第2层：存储着找到的所有可能的较小尺寸的对象
# 每一层中包含许多个可能的对象，这些对象都是由85个值所构成的。
# 第0-3个值是边框自身位置、大小信息（需要注意，值都是相对于原始图像的百分比形式）
# 第4个值是边框的置信度
# 第5-84（共计80个）表示80个置信度，对应classes中80种对象每种对象的可能性。
# ===========获取置信度较高的边框===========
# 置信度较高的边框相关的三个值：resultIDS、boxes、confidences
resultIDS = [] # 置信度较高的边框对应的分类在classes中的ID值
boxes = [] # 置信度较高的边框集合
confidences = [] # 置信度较高的边框的置信度
(H, W) = image.shape[:2]   #原始图像image的宽、高（辅助确定图像内各个边框的位置、大小）
for out in outs:  # 各个输出层（共3各层，逐层处理）
    for candidate in out:  #每个层中，包含几百个可能的候选框，逐个处理
        #每个candidate（共85个数值）包含三部分：
        # 第1部分：candidate[0:4]存储的是边框位置、大小（使用的是相对于image的百分比形式）
        # 第2部分：candidate[5]存储的是当前候选框的置信度（可能性）
        # 第3部分：第5-84个值(candidate[5:])，存储的是对应classes中每个对象的可能性（置信度）
        # 在第5-84个值中，找到其中最大值及对应的索引（位置）。两种情况：
        # 情况1：如果这个最大值大于0.5，说明当前候选框是最终候选框的可能性较大。
        # 保留当前可能性较大的候选框，留作后续处理。
        # 情况2：如果这个最大值不大于（小于等于）0.5，抛弃当前候选框。
        # 对应到程序上，不做任何处理。
        scores = candidate[5:]  # 先把第5-84个值筛选出来
        classID = np.argmax(scores)  # 找到其中最大值对应的索引（位置）
        confidence = scores[classID]  # 找到最大的置信度值（概率值）
        # 下面开始对置信度大于0.5的候选框进行处理。
        # 仅考虑置信度大于0.5的，小于该值的直接忽略（不做任何处理）
        if confidence > 0.5:
            # 获取候选框的位置、大小
            # 需要注意，位置、大小都是相对于原始图像image的百分比形式。
            # 因此，位置、大小通过将candidate乘以image的宽度、高度获取
            box = candidate[0:4] * np.array([W, H, W, H])  
            # 另外需要注意，candidate所表示的位置是矩形框（候选框）的中心点位置
            (centerX, centerY, width, height) = box.astype("int")
            # OpenCV中，使用左上角表示矩形框位置。
            # 将通过中心点获取左上角的坐标。
            #centerX，centerY是矩形框的中心点，通过他们计算出左上角坐标x,y
            x = int(centerX - (width / 2))   #x方向中心点-框宽度/2
            y = int(centerY - (height / 2))  #y方向中心点-框高度/2
            # 将当前可能性较高的候选框放入boxes中
            boxes.append([x, y, int(width), int(height)])
            # 将当前可能性较高的候选框对应的置信度confidence放入confidences内
            confidences.append(float(confidence))
            # 将当前可能性较高的候选框所对应的类别放入resultIDS中
            resultIDS.append(classID)
# 非极大值抑制，将众多重合的边框保留一个最关键的（去重处理）
indexes = cv2.dnn.NMSBoxes(boxes, confidences,0.5,0.4)
# indexes，所有可能边框的序号集合（需要注意，indexes表示的是boxes内的序号）
# ===========绘制边框===========
# 给每个分类随机分配一个颜色
classesCOLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8") 
# 绘制边框及置信度、分类
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        color = [int(c) for c in classesCOLORS[resultIDS[i]]]  #边框颜色
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        result = "{}: {:.0f}%".format(classes[resultIDS[i]], confidences[i]*100)
        cv2.putText(image, result, (x, y+35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)  
cv2.imshow('result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()