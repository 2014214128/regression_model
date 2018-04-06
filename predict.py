#! /usr/bin/env python
# coding:utf-8

# Copyright zhengguo <2402444249@qq.com> All Rights Reversed.
# This file is free software, distributed under the MIT License.

from numpy import *
from argparse import ArgumentParser


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--s', type=int,
                        dest='s', help='0 linearRegression, 1 logisticRegression',
                        metavar='S', required=True)
    parser.add_argument('--modelfile',
                        dest='modelfile', help='The trained model file',
                        metavar='MODELFILE', required=True)
    parser.add_argument('--predictfile',
                        dest='predictfile', help='The path of the training file',
                        metavar='PREDICTFILE', required=True)
    parser.add_argument('--delimiter',
                        dest='delimiter', help='Delimiters for predict files,the default is a space',
                        metavar='DELIMITER')
    parser.add_argument('--resultfile',
                        dest='resultfile', help='Forecast result document',
                        metavar='RESULTFILE', required=True)
    parser.add_argument('--threshold', type=float,
                        dest='threshold', help='Threshold of model classification',
                        metavar='THRESHOLD')

    return parser


def loadModel(modelfile):
    theta = []
    with open(modelfile, 'r') as f:
        for line in f:
            theta.append([float(line.strip())])
    return mat(theta)


def loadDataSet(predictfile, delitimer=' ', s=0):
    dataMat = []
    fr = open(predictfile)
    for line in fr.readlines():
        if delitimer == ' ':
            lineArr = line.strip().split()
        else:
            lineArr = line.strip().split(delitimer)
        if s == 0:
            dataMat.append([1.0, float(lineArr[0])])
        else:
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    return dataMat


def sigmiod(x):
    return 1.0 / (1.0 + exp(-1.0 * x))


def predict_logistic(X, theta, threshold=0.5):
    p = sigmiod(X.dot(theta)) >= threshold
    return p.astype('int')


def predict_linear(X, theta):
    return X.dot(theta)


def saveResult(resultfile, result):
    with open(resultfile, 'w') as f:
        arr = array(result)
        for i in range(result.shape[0]):
            f.write(str(arr[i][0]) + '\n')


if __name__ == '__main__':

    parser = build_parser()
    options = parser.parse_args()

    theta = loadModel(options.modelfile)
    dataMat = loadDataSet(options.predictfile, options.delimiter if not options.delimiter is None else ' ', options.s)

    if options.s == 0:
        result = predict_linear(mat(dataMat), theta)
    elif options.s == 1:
        result = predict_logistic(mat(dataMat), theta, options.threshold if not options.threshold is None else 0.5)

    saveResult(options.resultfile, result)
