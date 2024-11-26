# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:10:21 2021

@author: Administrator
"""

import cv2
import numpy as np
#========读取图像=============
image = cv2.imread("e.jpg")
#========根据原始图像构造一个背景，用于存放实例=============
(H, W) = image.shape[:2]
background = np.zeros((H, W, 3), np.uint8)
background[:] = (100, 100, 0)
#========读取类别信息=============
LABELS =open("object_detection_classes_coco.txt").read().strip().split("\n")
# 共计90个类别
# 分别为：person、bicycle、car、motorcycle等
#========加载模型、推理=============
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph.pb",
                                  "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
blob = cv2.dnn.blobFromImage(image, swapRB=True)
net.setInput(blob)
boxes, masks = net.forward(["detection_out_final", "detection_masks"])
# 返回100个候选框
# ---------boxes，候选框-----------
# boxes结构
# boxes.shape=(1,1,100,7)
# boxes[0, 0, :, 1]--------对应类别
# boxes[0, 0, :, 2]--------置信度
# boxes[0, 0, :, 3:7]--------候选框的位置（以相对于原始图像百分比形式表示）
# -----------masks，掩模（掩模、掩码）------------
# masks.shape=(100,90,15,15)
# 第1维masks[0]：共计100个候选框对应着100个模板mask(该维度的尺寸为100)
# 第2维masks[1]：模型中类的置信度（该维度的尺寸为90）
# 第3维和第4维masks[2:4]表示掩模，其尺寸为15× 15
# 使用掩模时，需要将其调整为与原始图像尺寸大小一致
#========实例分割处理=============
# 计算候选框的数量detectionCount
number = boxes.shape[2]   #该值是100（选出的候选框数量：100个）
# 遍历每一个候选框，将可能的实例进行标注
for i in range(number):
    # 获取类别名称
    classID = int(boxes[0, 0, i, 1])
    # 获取置信度
    confidence = boxes[0, 0, i, 2]
    # 考虑较大置信度的，将较小的忽略
    if confidence > 0.5:
        # 获取当前候选框的位置（将百分比形式转换为像素值形式）
        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])        
        (x1, y1, x2, y2) = box.astype("int")
        # 获取当前候选框(以切片形式从background内截取)
        box = background[y1: y2, x1: x2]
        # import  random  #供下一行random.randint使用
        # cv2.imshow("box" + str(random.randint(3,100)),box)  #测试各个候选框
        # 获取候选框的高度和宽度（可以通过box.shape计算，也可以通过坐标直接计算）
        # boxHeight, boxWidth= box.shape[:2] 
        boxHeight = y2 - y1
        boxWidth = x2 - x1
        # 获取当前的模板mask（单个masks的尺度15*15）
        mask = masks[i, int(classID)]
        # mask的大小为15*15像素大小，要调整到与候选框一致大小。
        mask = cv2.resize(mask, (boxWidth, boxHeight))
        # import  random  #供下一行random.randint使用
        # cv2.imshow("maska" + str(random.randint(3,100)),mask)  #测试各个候选框
        # 阈值处理,处理为二值形式
        rst, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        # import  random   #供下一行random.randint使用
        # cv2.imshow("maskb" + str(random.randint(3,100)),mask)  #测试各个实例
        #获取mask内的轮廓（实例）
        contours, hierarchy = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 设置随机颜色
        color = np.random.randint(0, 255, 3)
        color = tuple ([int(x) for x in color])  
        #设置为元组，整数
        # color是int64,需要转换为int【无法直接使用tuple(color)实现】
        # 绘制实例的轮廓（实心形式）
        cv2.drawContours(box,contours,-1,color,-1)
        # 输出对应的类别及置信度
        msg = "{}: {:.0f}%".format(LABELS[classID], confidence*100)
        cv2.putText(background, msg, (x1+50, y1 +45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
# 将识别结果和原始图像叠加在一起
result =cv2.addWeighted(image,0.2,background,0.8,0)
#========显示处理结果=============
cv2.imshow("original", image)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()