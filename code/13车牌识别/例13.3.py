# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 09:24:25 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""
# ==========================导入库==============================
import cv2
# from matplotlib import pyplot as plt
import numpy as np
import glob
# ==========================提取车牌函数==============================
def getPlate(image):
    rawImage=image.copy()
    # 去噪处理
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # 色彩空间转换（RGB-->GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Sobel算子（X方向边缘梯度）
    Sobel_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    absX = cv2.convertScaleAbs(Sobel_x)  # 映射到[0.255]内
    image = absX
    # 阈值处理
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    # 闭运算：先膨胀后腐蚀，车牌各个字符是分散的，让车牌构成一体
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX)
    # 开运算：先腐蚀后膨胀，去除噪声
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernelY)
    # 中值滤波：去除噪声
    image = cv2.medianBlur(image, 15)
    # 查找轮廓
    contours, w1 = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #测试语句，查看处理结果
    # image = cv2.drawContours(rawImage.copy(), contours, -1, (0, 0, 255), 3)
    # cv2.imshow('imagecc', image)
    #逐个遍历，将宽度>3倍高度的轮廓确定为车牌
    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        if weight > (height * 3):
            plate = rawImage[y:y + height, x:x + weight]    
    return plate
#======================预处理函数，图像去噪等处理=================
def preprocessor(image):
    # 图像去噪灰度处理
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # 色彩空间转换
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 阈值处理（二值化）   
    ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    #膨胀处理，让一个字构成一个整体（大多数字不是一体的，是分散的）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.dilate(image, kernel)
    return image
#===========拆分车牌函数，将车牌内各个字符分离==================
def splitPlate(image):
    # 查找轮廓，各个字符的轮廓
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    # 遍历所有轮廓
    for item in contours:
        rect = cv2.boundingRect(item)
        words.append(rect)
    # print(len(contours))  #测试语句：看看找到多少个轮廓
    #-----测试语句：看看轮廓效果-----
    # imageColor=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    # x = cv2.drawContours(imageColor, contours, -1, (0, 0, 255), 1)
    # cv2.imshow("contours",x)    
    #-----测试语句：看看轮廓效果-----
    # 按照x轴坐标值排序（自左向右排序）
    words = sorted(words,key=lambda s:s[0],reverse=False)    
    # 用word存放左上角起始点及长宽值
    plateChars = []
    for word in words:
        # 筛选字符的轮廓(高宽比在1.5-8之间，宽度大于3)
        if (word[3] > (word[2] * 1.5)) and (word[3] < (word[2] * 8)) and (word[2] > 3):
            plateChar = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            plateChars.append(plateChar)
    # 测试语句：查看各个字符
    # for i,im in enumerate(plateChars):
    #     cv2.imshow("char"+str(i),im)
    return plateChars 
#=================模板，部分省份，使用字典表示==============================
templateDict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
            10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',
            18:'J',19:'K',20:'L',21:'M',22:'N',23:'P',24:'Q',25:'R',
            26:'S',27:'T',28:'U',29:'V',30:'W',31:'X',32:'Y',33:'Z',
            34:'京',35:'津',36:'冀',37:'晋',38:'蒙',39:'辽',40:'吉',41:'黑',
            42:'沪',43:'苏',44:'浙',45:'皖',46:'闽',47:'赣',48:'鲁',49:'豫',
            50:'鄂',51:'湘',52:'粤',53:'桂',54:'琼',55:'渝',56:'川',57:'贵',
            58:'云',59:'藏',60:'陕',61:'甘',62:'青',63:'宁',64:'新', 
            65:'港',66:'澳',67:'台'}
# ==================获取所有字符的路径信息===================
def getcharacters():
    c=[]
    for i in range(0,67):
        words=[]
        words.extend(glob.glob('template/'+templateDict.get(i)+'/*.*'))
        c.append(words)
    return c
#=============计算匹配值函数=====================
def getMatchValue(template,image):
    #读取模板图像
    # templateImage=cv2.imread(template)   #cv2读取中文文件名不友好
    templateImage=cv2.imdecode(np.fromfile(template,dtype=np.uint8),1)
    #模板图像色彩空间转换，BGR-->灰度
    templateImage = cv2.cvtColor(templateImage, cv2.COLOR_BGR2GRAY)
    #模板图像阈值处理， 灰度-->二值
    ret, templateImage = cv2.threshold(templateImage, 0, 255, cv2.THRESH_OTSU)
    # 获取待识别图像的尺寸
    height, width = image.shape
    # 将模板图像调整为与待识别图像尺寸一致
    templateImage = cv2.resize(templateImage, (width, height))
    #计算模板图像、待识别图像的模板匹配值
    result = cv2.matchTemplate(image, templateImage, cv2.TM_CCOEFF)
    # 将计算结果返回
    return result[0][0]
# ===========对车牌内字符进行识别====================
#plates，要识别的字符集，
# 也就是从车牌图像“GUA211”中分离出来的每一个字符的图像"G","U","A","2","1","1"
#chars，所有字符的模板集合，也就是0-9，A-Z，京-台，每一个字符模板
def matchChars(plates,chars):
    results=[]   #存储所有的识别结果
    #最外层循环：逐个遍历要识别的字符。
    # 例如，逐个遍历从车牌图像“GUA211”中分离出来的每一个字符的图像
    # 如"G","U","A","2","1","1"
    # plateChar分别存储，"G","U","A","2","1","1"
    for plateChar in plates:#逐个遍历要识别的字符
        #bestMatch，存储的是待识别字符与每个特征字符的所有模板中最匹配的模板
        # 例如，待识别图像“G”，与所有的字符0-9，A-Z，京-台，每一个字符最匹配的模板
        bestMatch = []      #最佳匹配
        #中间层循环：针对模板内的字符，进行逐个遍历（每次循环针对一个特定的字符），
        #words 对应的是每一个字符（例如字符A）的所有模板
        for words in chars: #遍历字符。chars：所有模板，words：某个字符的所有模板
            #match，存储的是每个特征字符的所有匹配值
            # 例如：待识别图像“G”，与字符7的所有模板的匹配值
            match = []      #每个字符的匹配值
            #最内层循环：针对的是单个字符的所有模板，找到最佳的模板
            #  word对应的是单个模板
            for word in words:  #遍历模板。words：某个字符所有模板，word单个模板
                result = getMatchValue(word,plateChar)
                match.append(result)
            bestMatch.append(max(match))   #将每个字符模板的最佳匹配加入bestMatch
        i = bestMatch.index(max(bestMatch))  #i是最佳匹配的字符模板的索引值
        r = templateDict[i]    #r是单个待识别字符的识别结果
        results.append(r)   #将每一个分割字符的识别结果加入到results内
    return results   #返回所有的识别结果
# ================主程序=============
image = cv2.imread("gua.jpg")           #读取原始图像
cv2.imshow("original",image)            #显示原始图像
image=getPlate(image)                   #获取车牌
cv2.imshow('plate', image)              #测试语句：看看车牌定位情况
image=preprocessor(image)               #预处理
# cv2.imshow("imagePre",image)          #测试语句，看看预处理结果
plateChars=splitPlate(image)            #分割车牌，将每个字符独立出来
for i,im in enumerate(plateChars):      #逐个遍历字符
    cv2.imshow("plateChars"+str(i),im)  #显示分割的字符
chars=getcharacters()                   #获取所有模板文件（文件名）
results=matchChars(plateChars, chars)   #使用模板chars逐个识别字符集plates
results="".join(results)                #将列表转换为字符串
print("识别结果为：",results)            #输出识别结果
cv2.waitKey(0)                          #显示暂停
cv2.destroyAllWindows()                 #释放窗口    