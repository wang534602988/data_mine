# -*- coding: utf-8 -*-
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import GaussianNB

from dataProcess import out_data, write_data, write  # 导入dataProcess中的一些函数

TRAIN_FILE = 'adult.data'
TEST_FILE = 'adult.test'
TRAIN_OUT = 'newtrain.data'
TEST_OUT = 'newtest.data'
TRAIN_LABEL = 'labeltrain.data'
TEST_LABEL = 'labeltest.data'


# 将训练集与测试集的数据合并
def mix(train_x, train_y, test_x, test_y):
    return np.vstack((train_x, test_x)), np.vstack((train_y, test_y)).ravel()


# k折交叉验证
def k_clia(clf, x, y):
    test_accurcy = cross_val_score(clf, x, y, cv=5).mean()
    print("k折交叉验证正确率为：%f%%" % (test_accurcy * 100))
    return test_accurcy


# roc绘制
def roc(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    plot_roc_curve(clf, x_test, y_test)
    plt.savefig(str(clf) + '_roc.png')


# 一些常见评价参数与混淆矩阵，以及结果保存
def confusion_and_save(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    write_data(str(clf) + '.data', np.hstack((x_test, pred.reshape(-1, 1))))
    print('准确率：', accuracy_score(y_test, pred))
    print('精确率：', precision_score(y_test, pred))
    print('召回率：', recall_score(y_test, pred))
    print('混淆矩阵如下：')
    martix = confusion_matrix(y_test, pred)
    print(martix)
    return martix


# 随机参数优化
def random_param(clf, paramter, x, y, n_iter_search=20):
    random_search = RandomizedSearchCV(clf, param_distributions=paramter, n_iter=n_iter_search)
    random_search.fit(x, y)
    print(random_search.best_params_)
    print(random_search.best_score_)
    return random_search.best_params_, random_search.best_score_


# 遍历参数优化
def grid_param(clf, paramter, x, y, cv=5):
    search = GridSearchCV(clf, paramter, cv=cv)
    search.fit(x, y)
    print(search.best_params_, search.best_score_)
    return search.best_params_, search.best_score_


# 合并以上的一些评估方法
def metric(clf, train_x, train_y, test_x, test_y):
    x, y = mix(train_x, train_y, test_x, test_y)
    k_clia(clf, x, y)
    roc(clf, train_x, train_y.ravel(), test_x, test_y.ravel())
    confusion_and_save(clf, train_x, train_y.ravel(), test_x, test_y.ravel())


def logistic(train_x, train_y, test_x, test_y):
    time_start = time.time()  # 开始计时
    clf = LogisticRegression(solver='liblinear', max_iter=100)
    paramter = {
        'max_iter': (10, 100, 1000),
        # 'solver': ('liblinear', 'lbfgs', 'sag', 'newton-cg', 'saga')
    }
    print(clf)
    x, y = mix(train_x, train_y, test_x, test_y)
    grid_param(clf, paramter, x, y, 5)
    metric(clf, train_x, train_y, test_x, test_y)
    time_end = time.time()  # 结束计时
    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')


def random_forest(train_x, train_y, test_x, test_y):
    time_start = time.time()  # 开始计时
    clf = RandomForestClassifier(n_estimators=10)
    paramter = {"max_depth": [3, None],
                "criterion": ["gini", "entropy"],
                "min_samples_split": sp_randint(2, 11),
                "min_samples_leaf": sp_randint(1, 11),
                "max_features": sp_randint(1, 11),
                "bootstrap": [True, False],
                }  # 参数字典
    print(clf)
    metric(clf, train_x, train_y, test_x, test_y)
    time_end = time.time()  # 结束计时
    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')
    print('随机参数优化:')
    x, y = mix(train_x, train_y, test_x, test_y)
    random_param(clf, paramter, x, y, 20)


def gaussianNB(train_x, train_y, test_x, test_y):
    time_start = time.time()  # 开始计时
    clf = GaussianNB()
    print(clf)
    metric(clf, train_x, train_y, test_x, test_y)
    time_end = time.time()  # 结束计时
    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')


if __name__ == '__main__':
    write(TRAIN_FILE, TRAIN_OUT, TRAIN_LABEL)
    write(TEST_FILE, TEST_OUT, TEST_LABEL)
    train_x, train_y = out_data(TRAIN_FILE)
    # 该语句用于将训练数据集分解为训练集和测试集，由于已经有了测试集，故省去该步骤
    # train_x,test_x,train_y,test_y /
    # = train_test_split(train_x,train_y,test_size=0.2,random_state=0)
    test_x, test_y = out_data(TEST_FILE)
    logistic(train_x, train_y, test_x, test_y)
    random_forest(train_x, train_y, test_x, test_y)
    gaussianNB(train_x, train_y, test_x, test_y)
