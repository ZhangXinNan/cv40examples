# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 17:32:43 2021

@author: Administrator
"""


import numpy as np
import cv2
image=np.random.randint(0,256,(4,4),np.uint8)
print("原始数据：\n",image)
blob = cv2.dnn.blobFromImage(image,1,(4,2),0,True,crop=True)
print("直接裁剪1：\n",blob[0])
blob = cv2.dnn.blobFromImage(image,1,(2,4),0,True,crop=True)
print("直接裁剪2：\n",blob[0])
blob = cv2.dnn.blobFromImage(image,1,(4,2),0,True,crop=False)
print("不裁剪1：\n",blob[0])
blob = cv2.dnn.blobFromImage(image,1,(2,4),0,True,crop=False)
print("不裁剪2：\n",blob[0])










# import numpy as np
# import cv2
# image=np.random.randint(0,256,(4,4),np.uint8)
# print("原始数据：\n",image)
# blob = cv2.dnn.blobFromImage(image,1,(4,2),0,True,crop=True)
# print("直接裁剪：\n",blob[0])
# blob = cv2.dnn.blobFromImage(image,1,(2,2),0,True,crop=True)
# print("缩放裁剪：\n",blob[0])
# blob = cv2.dnn.blobFromImage(image,1,(3,2),0,True,crop=False)
# print("不裁剪：\n",blob[0])
