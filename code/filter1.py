from sklearn import feature_selection
import sys
import numpy as np
from sklearn.datasets import load_iris

data = load_iris().data # shape=(150,4)

#计算每个特征的方差
vars = [np.var(data[:,index]) for index in range(len(data[0]))]
print("特征选择之前所有的特征对应方差为：")
print(vars)
sele = feature_selection.VarianceThreshold(threshold=0.6)
dataNew = sele.fit_transform(data)#两个fit，这个会返回选择后的数据集
sele.fit(dataNew)
print("特征选择之后所有的特征对应方差（通过属性variances_得到）为：")
print(sele.variances_)