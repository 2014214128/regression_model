#! /usr/bin/env python
# coding:utf-8

# Copyright zhengguo <2402444249@qq.com> All Rights Reversed.
# This file is free software, distributed under the MIT License.

from numpy import *


class LinearRegression(object):
    def __init__(self, X, y):
        self.m = len(X)
        self.n = len(X[0])
        self.theta = ones((self.n, 1))
        self.X = mat(X)
        self.y = mat(y).transpose()

    def costFunction(self):
        h = dot(self.X, self.theta)
        J = 0.5 * (1.0/self.m) * sum(array((self.y - h)) ** 2)
        if isnan(J):
            return (inf)
        return J

    def gradientDescent(self, alpha):
        h = dot(self.X, self.theta)
        self.grad = transpose(self.X).dot(self.y - h)
        self.theta = self.theta + alpha * self.grad

    def stocGradientDescent(self, alpha):
        for i in range(self.m):
            h = dot(self.X[i], self.theta)
            error = self.y[i] - h
            self.theta = self.theta + alpha * self.X[i].transpose() * error

    def train(self, iterations=1000, alpha=0.01, method=0):
        for k in range(iterations):
            if method == 0:
                self.gradientDescent(alpha)
            elif method == 1:
                self.stocGradientDescent(alpha)

    def predict(self, threshold=0.5):
        return self.X.dot(self.theta)
