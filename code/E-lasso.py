#from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
# 回归估计器有linear_model.Lasso,分类估计器有linear_model.LogisticRegression和svm.LinearSVC
# SVMs和logistic-regression的API里,参数C控制特征的稀疏性,C越小要筛选剩下的特征越少.
# Lasso的API里,参数alpha越大要筛选剩下的特征越少
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)

lasso = Lasso(alpha=0.01).fit(X, y)

model = SelectFromModel(lasso, prefit=True)
X_new = model.transform(X)
print(X_new.shape)