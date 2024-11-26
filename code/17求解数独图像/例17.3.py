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
#=======函数：获取所有轮廓、数字的轮廓信息=========
def location(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     #灰度化
    ret,thresh = cv2.threshold(gray,200,255,1)      #阈值处理
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))      #核结构
    dilated = cv2.dilate(thresh,kernel)             #膨胀
    # 获取轮廓
    mode = cv2.RETR_TREE   #轮廓检测模式
    method = cv2.CHAIN_APPROX_SIMPLE #轮廓近似方法
    contours, hierarchy = cv2.findContours(dilated,mode,method)   #提取轮廓
    #　------------   提取小单元格（9*9=81个）  ---------------
    boxHierarchy = []   #小单元格的hierarchy信息    
    imgBox=img.copy()   #用于显示每一个单元格的轮廓
    for i in range(len(hierarchy[0])):      #针对外轮廓
        if hierarchy[0][i][3] == 0:         #判断：父轮廓是外轮廓的对象（小单元格）
            boxHierarchy.append(hierarchy[0][i])   #将该符合条件轮廓的hierarchy放入boxHierarchy
            imgBox=cv2.drawContours(imgBox.copy(),contours,i,(0,0,255)) #绘制单元格轮廓            
    cv2.imshow("boxes", imgBox)          #显示每一个小单元格的轮廓，测试用
    #　-------------  提取数字边框（定位数字） -----------------
    numberBoxes=[]   #所有数字的轮廓
    imgNum=img.copy()  #用于显示数字轮廓
    for j in range(len(boxHierarchy)):
        if boxHierarchy[j][2] != -1:  #符合条件的是包含数字的小方格
            numberBox=contours[boxHierarchy[j][2]]  #小单元内数字轮廓
            numberBoxes.append(numberBox)     #每个数字轮廓加入numberBoxes中
            x,y,w,h = cv2.boundingRect(numberBox)   #矩形包围框
            # 绘制矩形边框
            imgNum = cv2.rectangle(imgNum.copy(),(x-1,y-1),(x+w+1,y+h+1),(0,0,255),2)
    cv2.imshow("imgNum", imgNum)          #数字轮廓
    return contours,numberBoxes     #返回所有轮廓、数字轮廓
# ================函数：训练模型=================
def getModel():
    cols=15         #控制调整后图像列数
    rows=20         #控制调整图像后行数
    s=cols*rows     #控制调整后图像尺寸
    xt=[]
    for i in range(1,10):
        iTen=glob.glob('template/'+str(i)+'/*.*')
        trains=[]
        for number in iTen:
            train=cv2.imread(number)
            train=cv2.cvtColor(train,cv2.COLOR_BGR2GRAY)
            train=cv2.resize(train,(cols,rows))
            # ------------阈值处理----------------
            am=cv2.ADAPTIVE_THRESH_GAUSSIAN_C       #自适应方法adaptiveMethod
            tt=cv2.THRESH_BINARY                    #threshType阈值处理方式
            train = cv2.adaptiveThreshold(train,255,am,tt,11,2) 
            trains.append(train)
        xt.append(trains)
        
    x=np.array(xt)
    # 数据调整，将每个数字的尺寸由15*20调整为1*300（一行300个像素）
    train = x[:,:8].reshape(-1,s).astype(np.float32) 
    test = x[:,8:].reshape(-1,s).astype(np.float32) 
    # 分别为训练数据、测试数据分配标签（图像对应的实际值）
    k = np.arange(1,10)
    train_labels = np.repeat(k,8)[:,np.newaxis]
    test_labels = np.repeat(k,2)[:,np.newaxis]
    # 核心代码：初始化、训练、预测
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    ret,result,neighbours,dist = knn.findNearest(test,k=5)
    # 通过测试集校验准确率
    matches = (result.astype(np.int32)==test_labels)
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print( "当前使用KNN识别手写数字的准确率为:",accuracy )
    return knn
#====================函数：识别数独图像内的印刷体数字========================
def recognize(img,knn,contours,numberBoxes): 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     #灰度化
    # ------------阈值处理----------------
    am=cv2.ADAPTIVE_THRESH_GAUSSIAN_C      #自适应方法adaptiveMethod
    tt=cv2.THRESH_BINARY_INV               #threshType阈值处理方式
    thresh = cv2.adaptiveThreshold(gray,255,am,tt,11,2) 
    cols=15         #控制调整后图像列数
    rows=20         #控制调整图像后行数
    s=cols*rows     #控制调整后图像尺寸
    #----计算每个数字所占小单元大小----------
    height,width = img.shape[:2]
    box_h = height/9
    box_w = width/9
    # 初始化数独数组
    soduko = np.zeros((9, 9),np.int32)
    #=======识别原始数独图像内的数字，并据此构造对应的数组========        
    for nb in numberBoxes:        
        x,y,w,h = cv2.boundingRect(nb)   #获取数字的矩形包围框
        #  对提取的数字进行处理
        numberBox = thresh[y:y+h, x:x+w]
        # 尺寸调整，统一调整为15*20像素大小
        resized=cv2.resize(numberBox,(cols,rows))
        # 展开成一行1*300，符合knn格式
        sample = resized.reshape((1,s)).astype(np.float32)
        # knn识别
        retval, results, neigh_resp, dists = knn.findNearest(sample, k=5)        
        # 获取识别结果
        number = int(results[0][0])
        # 在原始数独图像上显示识别数字
        cv2.putText(img,str(number),(x+w-6,y+h-15), 3, 1, (255, 0, 0), 2, cv2.LINE_AA)            
        # 将数字存储在数组soduko中
        soduko[int(y/box_h)][int(x/box_w)] = number                   
    print("图像所对应的数独：")
    print(soduko)   #打印识别完已有数字的数独图像    
    cv2.imshow("recognize", img)   #显示识别结果       
    return soduko
#==================主程序=====================
original = cv2.imread('xt.jpg')
cv2.imshow("original",original)
contours,numberBoxes=location(original)
knn=getModel()
soduko=recognize(original,knn,contours,numberBoxes)
cv2.waitKey()
cv2.destroyAllWindows()
