# -*- coding: utf-8 -*-
"""
@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""
import cv2
import glob
import numpy as np
# ================函数：训练模型=================
def getModel():
    # step1:预处理
    # 主要工作：读取图像、预处理（色彩空间转换、大小调整、阈值处理）、处理为array
    cols=15         #控制调整后图像列数
    rows=20         #控制调整图像后行数
    s=cols*rows     #控制调整后图像尺寸
    data=[]   #存储所有数字的所有图像
    for i in range(1,10):
        iTen=glob.glob('template/'+str(i)+'/*.*')   # 某特定数字的所有图像的文件名
        num=[]      #临时列表，每次循环用来存储某一个数字的所有图像
        for number in iTen:    #逐个提取文件名
            image=cv2.imread(number)   #逐个读取文件，放入image中
            # 预处理：色彩空间变换
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # 预处理：大小调整
            image=cv2.resize(image,(cols,rows))
            # ------------预处理：阈值处理----------------
            ata=cv2.ADAPTIVE_THRESH_GAUSSIAN_C       #自适应方法adaptiveMethod
            tb=cv2.THRESH_BINARY                    #threshType阈值处理方式
            image = cv2.adaptiveThreshold(image,255,ata,tb,11,2) 
            num.append(image)  #把当前图像值放入num中
        data.append(num)  #把单个数字的所有图像放入data        
    data=np.array(data)
    # step2：划分数据集——划分为训练集和测试集
    train = data[:,:8]
    test = data[:,8:]
    # step3:塑形
    # 数据调整，将每个数字的尺寸由15*20调整为1*300（一行300个像素）
    train = train.reshape(-1,s).astype(np.float32) 
    test = test.reshape(-1,s).astype(np.float32) 
    # step 4：打标签。
    # 分别为训练数据、测试数据分配标签（图像对应的实际值）
    k = np.arange(1,10)
    train_labels = np.repeat(k,8)[:,np.newaxis]
    test_labels = np.repeat(k,2)[:,np.newaxis]
    # step5：使用KNN模块
    # 核心代码：初始化、训练、预测
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    ret,result,neighbours,dist = knn.findNearest(test,k=5)
    # step6：验证——通过测试集校验准确率
    matches = (result.astype(np.int32)==test_labels)
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print( "当前使用KNN识别手写数字的准确率为:",accuracy )
    # step7:返回训练好的模型
    return knn
#==================主程序=====================
knn=getModel()
