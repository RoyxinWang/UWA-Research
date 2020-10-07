from BasicFeatures import BasicFeatures
from ClusteringFeatures import ClusteringFeatures
from DatatypeFeatures import CategoricalFeatures
from ClassFeatures import ClassFeatures
from DistributionFeatures import DistributionFeatures
from MissingDataFeatures import MissingValuesFeatures
from AddedFeatures import AdFeatures

from sklearn.datasets import load_svmlight_file
import pandas as pd
import numpy as np


class MetaFeatures:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def calculate(self):
        list1 = self.get_basic_features()
        list2 = self.get_class_features()
        list3 = self.get_distribution_features()
        list4 = self.get_categorical_features()
        list5 = self.get_missing_values_features()
        list6 = self.get_clustering_features()
        list7 = self.get_adfeatures()
        raw = list1 + list2 + list3 + list4 + list5 + list6 + list7
        # norm = [float(i) / max(raw) for i in raw]
        return raw

    def get_basic_features(self):
        return BasicFeatures(self.X, self.y).value

    def get_clustering_features(self):
        return ClusteringFeatures(self.X, self.y).value

    def get_categorical_features(self):
        return CategoricalFeatures(self.X, self.y).value

    def get_class_features(self):
        return ClassFeatures(self.X, self.y).value

    def get_distribution_features(self):
        return DistributionFeatures(self.X, self.y).value

    def get_missing_values_features(self):
        return MissingValuesFeatures(self.X, self.y).value
    
    def get_adfeatures(self):
        return AdFeatures(self.X, self.y).value


def get_metafeatures(X_train, y_train):
    #X_train, y_train = load_svmlight_file(path)
    mat = X_train
    X = pd.DataFrame(mat)
    X.columns = range(len(X.columns))
    y = pd.DataFrame(y_train)
    y.columns = ['target']
    metafeatures = MetaFeatures(X, y).calculate()
    f = lambda x: x.item() if isinstance(x, np.generic) else x
    metafeatures = [f(i) for i in metafeatures]
    if type == "norm":
        return [float(i) / max(metafeatures) for i in metafeatures]
    return metafeatures

from sklearn.datasets import load_breast_cancer

iris = load_breast_cancer()
data, label = iris.data, iris.target

def load_data(File):
    return pd.read_csv(File,header=None,sep = ',',index_col=False)

# Import The Data
#original_data= load_data("E:/GENG5511 research/research/dataset/data_banknote_authentication.csv")

# Take a Quick Look at the Data Structure
#original_data.head()

# To store purely features data from dataset
#data = original_data.iloc[:, :4].values

# To store purely label from dataset
#label = original_data.iloc[:,-1]

pd.DataFrame(get_metafeatures(data, label)).to_csv('E:/GENG5511 research/research/dataset/123.csv', index=False)