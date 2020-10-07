from sklearn import feature_selection
import sys
import numpy as np
from sklearn.datasets import load_iris

data = load_iris().data  # shape=(150,4)
label = load_iris().target

# 指定评估标准
scoreFun = feature_selection.chi2

sele = feature_selection.SelectKBest(score_func=scoreFun,k=2)
dataNew = sele.fit_transform(data,label)
print("原始数据集每个特征对应得分：")
print(sele.fit(data,label).scores_)
print('特征选择之后对应特征的得分：')
print(sele.fit(dataNew,label).scores_)#每个特征对应的得分
print('对应P值：')
print(sele.pvalues_)#得分对应的P值