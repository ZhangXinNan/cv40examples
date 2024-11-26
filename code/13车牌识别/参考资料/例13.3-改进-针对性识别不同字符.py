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
    # cv2.imshow("imageT",image)   #测试语句，查看处理结果
    # 闭运算：先膨胀后腐蚀，车牌各个字符是分散的，让车牌构成一体
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX)
    # cv2.imshow("image1",image)    #测试语句，查看处理结果
    #闭运算： 先膨胀后腐蚀
    # kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX)
    # # cv2.imshow("image2",image)   #测试语句，查看处理结果
    #开运算：先腐蚀后膨胀
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernelY)
    # cv2.imshow("image3",image)
    # 中值滤波：去除噪声
    image = cv2.medianBlur(image, 15)
    # cv2.imshow("imageM",image)    #测试语句，查看处理结果
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
    word_images = []
    # 遍历所有轮廓
    for item in contours:
        word = []
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        word.append(x)
        word.append(y)
        word.append(weight)
        word.append(height)
        words.append(word)
    # print(len(contours))  #测试语句：看看找到多少个轮廓
    # x = cv2.drawContours(ximage, contours, -1, (0, 0, 255), 1)
    # cv2.imshow("contours",x)     #测试语句，看看车牌分割效果
    # 按照x轴坐标值排序（自左向右排序）
    words = sorted(words,key=lambda s:s[0],reverse=False)    
    # 用word存放左上角起始点及长宽值
    for word in words:
        # 筛选字符的轮廓
        if (word[3] > (word[2] * 1.5)) and (word[3] < (word[2] * 8)) and (word[2] > 3):
            splite_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            word_images.append(splite_image)
    # 测试语句：查看各个字符
    # for i,j in enumerate(word_images):  
    #     plt.subplot(1,7,i+1)
    #     plt.imshow(word_images[i],cmap='gray')
    return word_images 
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

# ==================获取汉字，用于匹配车牌的第1个汉字===================
def getChinese():
    c=[]
    for i in range(34,67):
        words=[]
        words.extend(glob.glob('template/'+templateDict.get(i)+'/*.*'))
        c.append(words)
    return c
# ================获取英文字母，匹配车牌的第2个字符==================
def getEnglish():
    e=[]
    for i in range(10,34):
        words=[]
        words.extend(glob.glob('template/'+templateDict.get(i)+'/*.*'))
        e.append(words)
    return e
# ===========获取英文字母和数字，用于匹配第3个字符开始的后续字符================
def getEngAndNum():
    en=[]
    for i in range(0,34):
        words=[]
        words.extend(glob.glob('template/'+templateDict.get(i)+'/*.*'))
        en.append(words)
    return en
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
# ============识别车牌中地区的简称=====================
def matchChinese(first,chineses):
    best_score = []
    for words in chineses:
        score = []
        for word in words:
            result = getMatchValue(word,first)
            score.append(result)
        best_score.append(max(score))
    i = best_score.index(max(best_score))
    # print(template[34+i])
    # r = templateDict.get(i)
    r=templateDict[34+i]
    return r  
# ================识别车牌中省份后面的字母=============
def matchEnglish(second,englishs):
    best_score = []
    for words in englishs:
        score = []
        for word in words:
            result = getMatchValue(word,second)
            score.append(result)
        best_score.append(max(score))
    i = best_score.index(max(best_score))
    # print(template[10+i])
    r = templateDict[10+i]
    return r
# ===========对车牌内第3到第7个字符进行识别（津G点号后面）====================
def matchOthers(others,engAndNums):
    results=[]
    for plateChar in others:
        best_score = []
        for words in engAndNums:
            score = []
            for word in words:
                result = getMatchValue(word,plateChar)
                score.append(result)
            best_score.append(max(score))
        i = best_score.index(max(best_score))
        r = templateDict[i]
        results.append(r)
    return results
# ================将不同位置上的识别结果连接一起=============
def getResults(r1,r2,r3):
    r1="".join(r1)
    r2="".join(r2)
    r3="".join(r3)
    results=r1+r2+r3
    return results
# ================主程序=============
image = cv2.imread("gua.jpg")       #读取原始图像
image=getPlate(image)                   #获取车牌
cv2.imshow('plate', image)            #测试语句：看看车牌定位情况
image=preprocessor(image)               #预处理
cv2.imshow("imagePre",image)          #测试语句，看看预处理结果
chineses=getChinese()                   #获取中文字符模板库(文件名)
englishs = getEnglish()                 #获取英文字符的模板库(文件名)
engAndNums = getEngAndNum()             #获取英文字符、数字的模板库(文件名)
plates=splitPlate(image)                #分割车牌，将每个字符独立出来
for i,im in enumerate(plates):      #逐个遍历字符
    cv2.imshow("plateChars"+str(i),im)  #显示分割的字符
r1=matchChinese(plates[0],chineses)     #识别第1个汉字省份简称
r2=matchEnglish(plates[1],englishs)     #识别省份后面的字母
r3=matchOthers(plates[2:],engAndNums)   #识别省份及字母后面字符
results=getResults(r1,r2,r3)            #将所有识别结果连接到一起
print("识别结果为：",results)            #输出识别结果
cv2.waitKey(0)                          #显示暂停
cv2.destroyAllWindows()                 #释放窗口    