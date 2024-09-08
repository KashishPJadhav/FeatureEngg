iimport numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

# Load dataset
dataset = pd.read_csv('diabetes.csv', header=None)

# Convert to numeric, marking errors as NaN
dataset = dataset.apply(pd.to_numeric, errors='coerce')

# Mark zero values as missing or NaN
dataset[[1, 2, 3, 4, 5]] = dataset[[1, 2, 3, 4, 5]].replace(0, np.NaN)

# Split dataset into inputs and outputs
values = dataset.values
X = values[:, 0:8]
y = values[:, 8]

# Check for NaN values
print(np.isnan(X).sum())
print(np.isnan(y).sum())

# Handle missing values by imputation
imputer = SimpleImputer(strategy='mean')  # Impute missing values with the mean of each column
X = imputer.fit_transform(X)

# Evaluate an LDA model on the dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=3, random_state=7, shuffle=True)
result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# Display the mean accuracy
print("Mean Accuracy: %.3f" % result.mean())
