# coding:utf-8
import random

from numpy import *
def loadDataSet():
    dataMat = [] ; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0 , float(lineArr[0]) , float(lineArr[1])])
        labelMat.append(int(lineArr[2])) #这样得到的是一行向量
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))
def gradAscent(dataMatIn , classLabels):
    dataMatrix = mat(dataMatIn) #一行一条数据
    labelMat = mat(classLabels).transpose() #转化为矩阵后转置
    m , n = shape(dataMatrix)
    alpha = 0.001 # 梯度上升(下降)的步长 w := w + a*偏f(w)
    maxCycles = 500 #迭代次数
    weights = ones((n,1)) # w 初始化 全为1的列向量
    for k in range(maxCycles):
        h = sigmoid((dataMatrix * weights)) #参数就是wT*x，dataMatrix的一行乘列向量weight
        error = (labelMat - h) #h是一列数据
        weights = weights + alpha * dataMatrix.transpose() * error  #全批量梯度下降法  ， dataMatrix.transpose() * error导数的结果,矩阵乘里面体现了全量的求和运算
    return weights                                                  #全批量梯度下降法如果在数据集特别大的时候效率很低，因为迭代都要遍历全部数据

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat , labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
        ax.scatter(xcord2,ycord2,s=30,c='green')
        x = arange(-3.0,3.0,0.1)
        y = (-weights[0]-weights[1]*x)/weights[2]
        ax.plot(x,y)
        plt.xlabel('X1');plt.ylabel('X2')
        plt.show()

def stocGradAscent0(dataMatrix , classLabels):
    m , n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n) #初始化权重
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights)) #点乘权重即 wTx 作为sigmoid的变量x
        error = classLabels[i] - h #和目标差距
        weights - weights + alpha*error*dataMatrix[i] #随机梯度上升
    return weights




#随机梯度上升（下降）法 改进，确定最佳回归系数
def stocGradAscent1(dataMatrix , classLabels , numIter = 150):
    m , n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m) #生成数组 [0 , ... ,m-1]
        for i in range(m):
            aplha = 4 / (1.0 + j + i) + 0.0001 # a动态
            randIndex = int(random.uniform(0,len(dataIndex))) #生成一个随机索引
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + aplha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex]) #使用完之后删除这条数据
        return weights


if __name__ == '__main__':
        dataArr , labelMat = loadDataSet()
        weights = stocGradAscent1(array(dataArr),labelMat,500)
        plotBestFit(weights)
        # print(__name__)