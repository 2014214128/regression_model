#! /usr/bin/env python
# coding:utf-8

# Copyright zhengguo <2402444249@qq.com> All Rights Reversed.
# This file is free software, distributed under the MIT License.

from argparse import ArgumentParser
from LogisticRegression import LogisticRegression
from LinearRegression import LinearRegression
from plotAndFit import *


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--s', type=int,
                        dest='s', help='0 linearRegression, 1 logisticRegression',
                        metavar='S', required=True)
    parser.add_argument('--trainfile',
                        dest='trainfile', help='The path of the training file',
                        metavar='TRAINFILE', required=True)
    parser.add_argument('--delimiter',
                        dest='delimiter', help='Delimiters for training files,the default is a space',
                        metavar='DELIMITER')
    parser.add_argument('--modelfile',
                        dest='modelfile', help='Output model file',
                        metavar='MODELFILE')
    parser.add_argument('--method', type=int,
                        dest='method',
                        help='0 gradientAscent(or gradientDescent) , 1 stocGradientAscent(or stocGradientDescent)',
                        metavar='METHOD')
    parser.add_argument('--iterations', type=int,
                        dest='iterations', help='Number of iterations of training',
                        metavar='ITERATIONS')
    parser.add_argument('--alpha', type=int,
                        dest='alpha', help='learning rate',
                        metavar='ALPHA')
    parser.add_argument('--threshold', type=float,
                        dest='threshold', help='Threshold of model classification',
                        metavar='THRESHOLD')
    return parser


def loadDataSet(trainfile, delitimer=' ', s=0):
    dataMat = []
    labelMat = []
    fr = open(trainfile)
    for line in fr.readlines():
        if delitimer == ' ':
            lineArr = line.strip().split()
        else:
            lineArr = line.strip().split(delitimer)
        if s == 0:
            dataMat.append([1.0, float(lineArr[0])])
            labelMat.append(float(lineArr[1]))
        elif s == 1:
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def saveModel(modelfile, theta):
    with open(modelfile, 'w') as f:
        arr = array(theta)
        n = theta.shape[0]
        for i in range(n):
            f.write(str(arr[i][0]) + '\n')


def process_result(res, options, dataMat, labelMat):
    print('theta:')
    print(res.theta)
    print('cost:')
    print(res.costFunction())

    if not options.modelfile is None:
        saveModel(options.modelfile, res.theta)

    if options.s == 0:
        plotBestFit_linear(dataMat, labelMat, res.theta)
    elif options.s == 1:
        plotBestFit_logistic(dataMat, labelMat, res.theta)
        p = res.predict(options.threshold if not options.threshold is None else 0.5)
        print('Train Accuracy: %f%%' % ((float(res.y[where(p == res.y)].size) / float(res.y.size)) * 100.0))


if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()

    dataMat, labelMat = loadDataSet(options.trainfile, options.delimiter if not options.delimiter is None else ' ',
                                    options.s)

    if options.s == 0:
        res = LinearRegression(dataMat, labelMat)
    else:
        res = LogisticRegression(dataMat, labelMat)

    res.train(options.iterations if not options.iterations is None else 1000,
              options.alpha if not options.alpha is None else 0.01, options.method if not options.method is None else 0)

    process_result(res, options, dataMat, labelMat)
