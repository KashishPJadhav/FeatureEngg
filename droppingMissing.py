from pandas import read_csv
import numpy
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
dataset = read_csv('diabetes.csv', header=None)
print(dataset.shape)
print(dataset.head(5))

# mark zero values as missing or NaN
dataset[[0,1,2,3,4,5,6]] = dataset[[0,1,2,3,4,5,6]].replace(0, numpy.NaN)


dataset_without_null = read_csv('diabetes.csv', header=None)

# mark zero values as missing or NaN
dataset_without_null = dataset.replace(0, numpy.NaN)

# drop rows with missing values
dataset_without_null.dropna(inplace=True)
# split dataset into inputs and outputs
values = dataset_without_null.values
X = values[:,0:8]
y = values[:,8]
print(X)
dataset_without_null.iloc[:,0:8]=X
print(dataset_without_null)