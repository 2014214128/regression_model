#! /usr/bin/env python
# coding:utf-8

# Copyright zhengguo <2402444249@qq.com> All Rights Reversed.
# This file is free software, distributed under the MIT License.

from numpy import *


def plotBestFit_linear(dataMat, labelMat, theta):
    import matplotlib.pyplot as plt
    dataArr = array(dataMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataArr[:, 1], array(labelMat), s=30, c='red', marker='s')
    x1_min, x1_max = dataArr[:, 1].min(), dataArr[:, 1].max()
    x = arange(x1_min, x1_max, 0.1)
    y = theta[0] + theta[1] * x
    ax.plot(x, transpose(y))
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def plotBestFit_logistic(dataMat, labelMat, theta):
    import matplotlib.pyplot as plt
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x1_min, x1_max = dataArr[:, 1].min(), dataArr[:, 1].max()
    x = arange(x1_min, x1_max, 0.1)
    y = (-theta[0] - theta[1] * x) / theta[2]
    ax.plot(x, transpose(y))
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
