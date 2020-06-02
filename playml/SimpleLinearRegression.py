#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : SimpleLinearRegression.py
# @Author: Fuxitong
# @Date  : 2020/5/31
# @Desc  :
import numpy as np


class SimpleLinearRegression1:

    # 参数学习 与 我之前写的kNN不同 我不需要存储 训练集和数据集

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, "目前我这个类只能接受一纬向量"
        assert len(x_train) == len(y_train), "训练集和数值的空间大小必须一样"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0

        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """给定一组数据 返回一组结果"""
        assert x_predict.ndim == 1, \
            "目前我这个类只能接受一纬向量"
        assert self.a_ is not None and self.b_ is not None, "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给定单个待预测数据x，返回x的预测结果值"""
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"


class SimpleLinearRegression2:

    # 尝试用numpy 的.dot 替换 for循环


    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, "目前我这个类只能接受一纬向量"
        assert len(x_train) == len(y_train), "训练集和数值的空间大小必须一样"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (y_train - y_mean).dot(x_train - x_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """给定一组数据 返回一组结果"""
        assert x_predict.ndim == 1, \
            "目前我这个类只能接受一纬向量"
        assert self.a_ is not None and self.b_ is not None, "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给定单个待预测数据x，返回x的预测结果值"""
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression2()"

class SimpleLinearRegression:

    def __init__(self):
        """结合一下 没啥改动"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1, "目前我这个类只能接受一纬向量"
        assert len(x_train) == len(y_train), "训练集和数值的空间大小必须一样"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        self.a_ = (x_train - x_mean).dot(y_train - y_mean) / (x_train - x_mean).dot(x_train - x_mean)
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
            "目前我这个类只能接受一纬向量"
        assert self.a_ is not None and self.b_ is not None, "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给定单个待预测数据x，返回x的预测结果值"""
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression()"