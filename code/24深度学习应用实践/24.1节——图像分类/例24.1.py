# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 14:16:58 2021

@author: Administrator
"""

import numpy as np
import cv2
#=======读取原始图像=============
image=cv2.imread("tower.jpg")
#=======调用模型=============
# 依次执行四个函数：
# readNetFromeCaffe/blogFromImage/setInput/forward
config='model/bvlc_googlenet.prototxt'
model='model/bvlc_googlenet.caffemodel'
net = cv2.dnn.readNetFromCaffe(config, model)
#与readnet不同，需要注意参数的先后顺序
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
net.setInput(blob)
prob = net.forward()
#=======读取分类信息============
classes =  open('model/label.txt', 'rt').read().strip().split("\n")
#=======确定分类所在行============
rowIndex = np.argsort(prob[0])[::-1][0]
#=======绘制输出结果============
result = "result: {}, {:.0f}%".format(classes[rowIndex],prob[0][rowIndex]*100)
cv2.putText(image, result, (25, 45),  cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)
#====显示原始输入图像======
cv2.imshow("result",image)
cv2.waitKey()
cv2.destroyAllWindows()

