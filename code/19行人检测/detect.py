"""
@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import numpy as np
import cv2


# ==============粗筛，筛选出包含人的矩形框====================
def detect(image):
    # 初始化我们的行人检测器
    hog = cv2.HOGDescriptor()   #初始化方向梯度直方图描述子
    #设置支持向量机(Support Vector Machine)使得它成为一个预先训练好的行人检测器
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  
    (rects, weights) = hog.detectMultiScale(image, 
                                winStride=(4, 4),padding=(8, 8), scale=1.05)
    return rects,weights
# ===============在图像上绘制出矩形框===================
#遍历每一个矩形框，将之绘制在图像上
def draw(orig,rects):
    for (x, y, w, h) in rects:  
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
#=============non max suppression，非极大值抑制处理====================
def NMS(boxes,  overlapThresh,weights):
    # 数据类型，非常关键，因为要进行除法运算。【3/5.0=0.6】VS【3/5=0】
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # 被选中的边框集，初始化为空
    pick = []
    # 获取边框的左上角(x1,y1),右下角(x2,y2)
    # 需要注意边框值依次为(x1,y1,w,h),w是宽度，h是高度
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x2 =  x1 + w
    y2 =  y1 + h
    area = w * h    
    # 矩形框那么多？先选谁，后选谁？如果有权重当然依靠权重
    # 这里没有权重，以y2为依据，也就是矩形框右下角点的y值，谁大先选谁
    # 也可以试试用面积或其他x1,x2,y1等可能想到的其他值
    idxs = y2  
    # idxs = weights  #权重不一定是最好的
    # 获取idxs的排序索引-->(最小值索引,....,中间值索引,...,最大值索引)
    idxs = np.argsort(idxs)

    # 循环：每次选择一个矩形框x到pick内，同时在原有矩形框集中删除x及其相似框
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        #==========得到每个矩形和当前选中矩形的交叉部分==============
        # 下面调整矩形，将其他矩形进行调整，保证：
        # 左上角在当前框外的，调整为当前矩形左上角
        # 左上角在当前框内的，保持不变
        # 右下角较在当前框外的，调整到当前框内
        # 右下角较在当前框内的，保持不变
        # 获取（每个矩形框左上角x值，当前选中矩形左上角x值）中的较小值
        # 获取（每个矩形框左上角y值，当前选中矩形左上角y值）中的较小值
        # 获取（每个矩形框右下角x值，当前选中矩形右下角x值）中的较小值
        # 获取（每个矩形框右下角y值，当前选中矩形右下角y值）中的较小值        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # 计算每个调整后矩形的宽度和高度
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # 计算其他矩形核当前矩形的面积比
        overlap = (w * h) / area[idxs[:last]]
        # 将面积比超过一定比例(二者重合过多的)删除
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # 将当前选中的矩形框返回
    return boxes[pick].astype("int")
# =============主程序=====================
original = cv2.imread("nms.jpg")             #主函数
nms = original.copy()                     #获取一个备份，对比非极大值抑制效果
rects,weights=detect(original)                  #获取行人对应的矩形框
print(rects.shape)
draw(original,rects)                    #绘制矩形框
pick = NMS(rects, 0.5,weights)    #应用非极大抑制方法
draw(nms,pick)              #绘制矩形框
cv2.imshow("original", original)     #显示原始效果
cv2.imshow("NMS", nms)   #显示非极大值抑制效果
cv2.waitKey(0)
cv2.destroyAllWindows()