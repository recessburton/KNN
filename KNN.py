#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Copyright (C),2014-2015, YTC, www.bjfulinux.cn
Created on  2015-04-06 15:04

@author: ytc recessburton@gmail.com
@version: 1.0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
'''

#P17#

from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator

#初始化数据#
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#KNN算法函数#
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #此处shape的值为(4,2)，shape[0]为4.
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #以4行一列的方式重复inX，此处为[1,1]，然后每个元素和group相减.
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1) #每个点求和，即每列求和.
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    
    classCount = {}   
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #类似于计数排序，每出现一次lebel得一票,此处投票结果:{'A': 2, 'B': 1}.
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True) #在classcount迭代中按照票数排序.
    return sortedClassCount[0][0]

#将训练文本转换为标准矩阵#
def file2matrix(filename):
    fr = open(filename) #打开文件
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]  #提取前三列约会数据
        labelindex = {'didntLike':'1','smallDoses':'2','largeDoses':'3'}    #用于从字符串类标签到数字类标签映射的字典
        classLabelVector.append(int(labelindex[listFromLine[-1]]))  #从字符串类标签到数字类标签的映射
        index += 1  #偏移指针，处理下一行。python没有C样式的for，所以每次的“善后工作”不能做更多的事。
    return returnMat, classLabelVector

#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)    #最小值行向量
    maxVals = dataSet.max(0)    #最小值行向量
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]        #样本总数
    normDataSet = dataSet - tile(minVals, (m,1))    #以m行1列的方式重复minval行向量来构造被减矩阵，用原始值减，得到偏差
    normDataSet /= tile(ranges, (m,1))             #偏差除以范围，得到0-1的归一化样本值 
    return normDataSet


#算法测试#
#group,labels = createDataSet()
#print "该点的类型为：",classify0([1,1], group, labels, 3)
datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
datingDataMat = autoNorm(datingDataMat)
#print datingDataMat,datingLabels[0:20]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0], datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()
