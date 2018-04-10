#! /usr/bin/env python
# coding:utf-8

# Copyright zhengguo <2402444249@qq.com> All Rights Reversed.
# This file is free software, distributed under the MIT License.

from numpy import *


class LogisticRegression(object):
    def __init__(self, X, y):
        self.m = len(X)
        self.n = len(X[0])
        self.theta = ones((self.n, 1))
        self.X = mat(X)
        self.y = mat(y).transpose()

    def sigmiod(self, x):
        return 1.0 / (1.0 + exp(-1.0 * x))

    def costFunction(self):
        h = self.sigmiod(dot(self.X, self.theta))
        J = -1.0 * (log(h).T.dot(self.y) + log(1 - h).T.dot(1 - self.y))
        if isnan(J[0]):
            return (inf)
        return J[0]

    def gradientAscent(self, alpha):
        h = self.sigmiod(dot(self.X, self.theta))
        self.grad = transpose(self.X).dot(self.y - h)
        self.theta = self.theta + alpha * self.grad

    def stocGradientAscent(self, alpha):
        for i in range(self.m):
            h = self.sigmiod(dot(self.X[i], self.theta))
            error = self.y[i] - h
            self.theta = self.theta + alpha * self.X[i].transpose() * error

    def train(self, iterations=1000, alpha=0.01, method=0):
        for k in range(iterations):
            if method == 0:
                self.gradientAscent(alpha)
            elif method == 1:
                self.stocGradientAscent(alpha)

    def predict(self, threshold=0.5):
        p = ((self.sigmiod(self.X.dot(self.theta))) >= threshold)
        return (p.astype('int'))
