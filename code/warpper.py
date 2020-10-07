from sklearn import datasets
from sklearn.feature_selection import RFE  #RFECV:带交叉验证的递归特征移除
from sklearn.linear_model import LogisticRegression

dataset =datasets.load_iris() # laod iris dataset
model = LogisticRegression() # build logistic regression model
rfe = RFE(model,2,step=1) # limit number of variables to three
rfe = rfe.fit(dataset.data,dataset.target)
print('要选择的特征数：')
print(rfe.n_features_)
print('对应特征的标记（是否被选中）：')
print(rfe.support_)
print('对应特征的优先级（1表示被选中）：')
print(rfe.ranking_)
print('所用的评估器：')
print(rfe.estimator_)