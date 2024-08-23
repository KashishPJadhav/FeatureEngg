import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Load dataset
dataset = pd.read_csv('diabetes.csv', header=None)

# Print the shape and first 5 rows of the dataset
print("Original Dataset Shape:", dataset.shape)
print("First 5 Rows of Original Dataset:")
print(dataset.head(5))

# Check for non-numeric data
print("\nData Types Before Conversion:")
print(dataset.dtypes)

# Mark zero values as missing or NaN
dataset.replace(0, np.NaN, inplace=True)

# Convert all data to numeric, forcing non-numeric data to NaN
dataset = dataset.apply(pd.to_numeric, errors='coerce')

# Check data types after conversion
print("\nData Types After Conversion:")
print(dataset.dtypes)

# Drop rows with missing values
dataset_without_null = dataset.dropna()
print("\nDataset After Dropping Rows with NaN Values:")
print(dataset_without_null.head())

# Split dataset into inputs and outputs
values = dataset_without_null.values
X = values[:, 0:8]
y = values[:, 8]

# Print transformed data for verification
print("\nTransformed X After Dropping NaNs:")
print(X)

# Impute missing values with mean column values
imputer = SimpleImputer(strategy="mean")
transformed_X_mean = imputer.fit_transform(X)
print("\nDataset After Imputation with Mean:")
dataset_with_mean = dataset.copy()
dataset_with_mean.iloc[:, 0:8] = imputer.transform(dataset.iloc[:, 0:8])
print(dataset_with_mean.head())

# Impute missing values with median column values
imputer = SimpleImputer(strategy="median")
transformed_X_median = imputer.fit_transform(X)
print("\nDataset After Imputation with Median:")
dataset_with_median = dataset.copy()
dataset_with_median.iloc[:, 0:8] = imputer.transform(dataset.iloc[:, 0:8])
print(dataset_with_median.head())

# Impute missing values with most frequent value
imputer = SimpleImputer(strategy="most_frequent")
transformed_X_most_frequent = imputer.fit_transform(X)
print("\nDataset After Imputation with Most Frequent Value:")
dataset_with_most_frequent = dataset.copy()
dataset_with_most_frequent.iloc[:, 0:8] = imputer.transform(dataset.iloc[:, 0:8])
print(dataset_with_most_frequent.head())

# Impute missing values with a constant value (0)
imputer = SimpleImputer(strategy="constant", fill_value=0)
transformed_X_constant_0 = imputer.fit_transform(X)
print("\nDataset After Imputation with Constant Value (0):")
dataset_with_constant_0 = dataset.copy()
dataset_with_constant_0.iloc[:, 0:8] = imputer.transform(dataset.iloc[:, 0:8])
print(dataset_with_constant_0.head())

# Impute missing values with a constant value (1)
imputer = SimpleImputer(strategy="constant", fill_value=1)
transformed_X_constant_1 = imputer.fit_transform(X)
print("\nDataset After Imputation with Constant Value (1):")
dataset_with_constant_1 = dataset.copy()
dataset_with_constant_1.iloc[:, 0:8] = imputer.transform(dataset.iloc[:, 0:8])
print(dataset_with_constant_1.head())

model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=3, shuffle=False,random_state=None)

# evaluate for mean replacement strategy
result = cross_val_score(model, transformed_X_mean, y, cv=kfold, scoring='accuracy')
print(" Accuracy for mean replacement  strategy          :- " + str(result.mean()))

# evaluate for median replacement strategy
result = cross_val_score(model, transformed_X_median, y, cv=kfold, scoring='accuracy')
print(" Accuracy for median replacement  strategy        :- " + str(result.mean()))

# evaluate for most_frequent replacement strategy
result = cross_val_score(model, transformed_X_most_frequent, y, cv=kfold, scoring='accuracy')
print(" Accuracy for most_frequent replacement  strategy :- " + str(result.mean()))

# evaluate for constant=0 replacement strategy
result = cross_val_score(model, transformed_X_constant_0, y, cv=kfold, scoring='accuracy')
print(" Accuracy for constant=0 replacement  strategy    :- " + str(result.mean()))

# evaluate for constant - 0 replacement strategy
result = cross_val_score(model, transformed_X_constant_1, y, cv=kfold, scoring='accuracy')
print(" Accuracy for constant=1 replacement  strategy    :- " + str(result.mean()))

