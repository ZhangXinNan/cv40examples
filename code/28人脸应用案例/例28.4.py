# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 20:20:57 2021

@author: Administrator
"""


import cv2
# =====模型初始化========
# 模型(网络模型/预训练模型):face/age/gender(脸、年龄、性别)
faceProto = "model/opencv_face_detector.pbtxt"
faceModel = "model/opencv_face_detector_uint8.pb"
ageProto = "model/deploy_age.prototxt"
ageModel = "model/age_net.caffemodel"
genderProto = "model/deploy_gender.prototxt"
genderModel = "model/gender_net.caffemodel"
# 加载网络
ageNet = cv2.dnn.readNet(ageModel, ageProto)  #年龄
genderNet = cv2.dnn.readNet(genderModel, genderProto)  #性别
faceNet = cv2.dnn.readNet(faceModel, faceProto)  # 人脸
# ============变量初始化============== 
# 年龄段和性别
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', 
           '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
mean = (78.4263377603, 87.7689143744, 114.895847746)   #模型均值
# ========自定义函数，获取人脸包围框===============
def getBoxes(net, frame):
    frameHeight, frameWidth = frame.shape[:2]  # 获取高度、宽度
    # 将图像（帧）处理为DNN可以接收的格式
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), 
                                 [104, 117, 123], True, False)
    # 调用网络模型，检测人脸
    net.setInput(blob)
    detections = net.forward()  
    # faceBoxes存储检测到的人脸
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  #筛选一下，将置信度大于0.7侧保留，其余不要了            
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])  # 人脸框的坐标
            # 绘制人脸框
            cv2.rectangle(frame, (x1, y1), (x2, y2), 
                          (0, 255, 0), int(round(frameHeight / 150)),6)  
    # 返回绘制了人脸框的帧frame、人脸包围框faceBoxes
    return frame, faceBoxes
# ==========循环读取每一帧，并处理=========
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   #装载摄像头
while True:
    # 读一帧
    _, frame = cap.read()
    # 做个镜像处理，左右互换(避免和照镜子一样是反的)
    # frame = cv2.flip(frame, 1)
    # 调用函数getFaceBox，获取人脸包围框、绘制人脸包围框（可能多个）
    frame, faceBoxes = getBoxes(faceNet, frame)
    if not faceBoxes:  #没有人脸时检测下一帧，后续循环操作不再继续。
        print("当前帧内不存在人脸")
        continue
    #  遍历每一个人脸包围框
    for faceBox in faceBoxes:
        # 处理frame，将其处理为符合DNN输入的格式
        blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227),mean)
        # 调用模型，预测性别
        genderNet.setInput(blob)   
        genderOuts = genderNet.forward()   
        gender = genderList[genderOuts[0].argmax()]   
        # 调用模型，预测年龄
        ageNet.setInput(blob)
        ageOuts = ageNet.forward()
        age = ageList[ageOuts[0].argmax()]
        # 格式化文本（年龄、性别）
        result = "{},{}".format(gender, age)
        # 输出性别和年龄
        cv2.putText(frame, result, (faceBox[0], faceBox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                   cv2.LINE_AA)         
        # 显示性别、年龄
        cv2.imshow("result", frame)
    # 按下Esc键，退出程序
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows() 
cap.release()
 