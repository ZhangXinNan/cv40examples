# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 17:32:53 2021

@author: Administrator
"""

import cv2
# ===============身体部位与姿势对============================
# 定义身体部位
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
# 姿势对
POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
#========加载模型、推理=============
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
# 处理实时视频
cap = cv2.VideoCapture(0)
while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    H, W =frame.shape[:2]    
    blob = cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    print(out.shape)
    # out[0]:图像索引。
    # out[1]: 关键点的索引。关键点热力图和部件亲和力图的置信度
    # out[2]: 第3维是输出图的高度。
    # out[3]: 第4个维度是输出图的宽度。
    #========核心步骤1：确定关键部位（关键点）=============
    out = out[:, :19, :, :]  #仅仅需要前19个（0~18）
    outH = out.shape[2]    #out的高度height
    outW = out.shape[3]    #out的宽度width
    points = []              #关键点
    print(points)
    for i in range(len(BODY_PARTS)):
        #身体对应部位的热图切片
        heatMap = out[0, i, :, :]
        # 取最值 
        _, confidence, _, point = cv2.minMaxLoc(heatMap)
        # 将out中关键点映射到原始图像image上
        px , py = point[:2]    
        x = ( px / outW ) * W
        y = ( py / outH ) * H
        # 仅将置信度大于0.2的关键点保留,其余的值为“None”。
        # 这里需要额外注意，不是仅仅保留置信度大于0.2的，同时将小于0.2的值设置为None
        # 后续判断需要借助None完成
        points.append((int(x), int(y)) if confidence > 0.2 else None)
    # print(points)   #观察一下points的情况，包含点和None两种值    
    # ========核心步骤2：绘制可能的姿势对================
    for posePair in POSE_PAIRS:   #逐个判断姿势对是否存在
        print("=============")
        partStart,partEnd = posePair[:2]        #取出姿势对中的两个关键点（关键部位）
        idStart = BODY_PARTS[partStart]         #取出姿势对中第1个关键部位的索引值
        idEnd = BODY_PARTS[partEnd]             #取出姿势对中第2个关键部位的索引值
        print(partStart,partEnd,idStart,idEnd,points[idStart] , points[idEnd])
        # 判断当前姿势对中的两个部位是否被检测到，如果检测到，将其绘制出来
        # 通过判断当前姿势对中的两个关键部位是否在points中实现
        if points[idStart] and points[idEnd]:    
            cv2.line(frame, points[idStart], points[idEnd], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idStart], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idEnd], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    # ==========显示最终结果===================
    cv2.imshow('result', frame)
cv2.destroyAllWindows()