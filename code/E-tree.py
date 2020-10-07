from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)

clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
print(clf.feature_importances_)  

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
print(X_new.shape) 

#Output:
#(150, 4)
#[ 0.04983538  0.04096076  0.56077362  0.34843023]
#(150, 2)