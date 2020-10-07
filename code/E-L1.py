from sklearn.svm import LinearSVC
# 回归估计器有linear_model.Lasso,分类估计器有linear_model.LogisticRegression和svm.LinearSVC
# SVMs和logistic-regression的API里,参数C控制特征的稀疏性,C越小要筛选剩下的特征越少.
# Lasso的API里,参数alpha越大要筛选剩下的特征越少
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)

model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print(X_new.shape)

#Output:
#(150, 4)
#(150, 3)