# coding=utf-8
from numpy import *
import operator
from os import listdir
import matplotlib
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group , labels

# classify 调用方式 classify0([0,0],group,labels,3)
def classify0(inX , dataSet , labels , k):
    dataSetSize = dataSet.shape[0] #数据维度
    #计算距离
    # tile函数 将inX 上下重复dataSetSize次,左右重复1次
    diffMat = tile(inX , (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2 #指数运算
    sqDistances = sqDiffMat.sum(axis=1) #axis=1 横轴相加
    distances = sqDistances ** 0.5 #开方
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel , 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip() #去掉回车
        listFromLine = line.split('\t') #tab分割
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1])) #python支持负索引，-1为倒数第一个元素
        index += 1
    return returnMat , classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals , (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet , ranges , minVals

def datingClassTest():
    hoRatio = 0.20
    datingDataMat , datingLabels = file2matrix('datingTestSet2.txt')
    normMat , ranges , minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # testMat[0:2, 1:3] 第0 1行的 1 2列
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]) : errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount

def classifyPerson():
    resultList = ['not like' , 'small like' , 'large like']
    percentTats = float(raw_input("percent of time spent playng games?:"))
    ffMiles = float(raw_input("fly miles earned per year?:"))
    iceCream = float(raw_input("liters of ice cream consumed per year?:"))
    datingDataMat , datingLabels = file2matrix('datingTestSet2.txt')
    normMat , ranges , minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles , percentTats , iceCream])
    classifyRes = classify0((inArr - minVals) / ranges , normMat , datingLabels , 3)
    print "You attitude to this person is:" , resultList[classifyRes - 1]








