# Load libraries
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# Create feature matrix with categorical feature
X = np.array([[0, 2.10, 1.45], 
              [1, 1.18, 1.33], 
              [0, 1.22, 1.27],
              [1, -0.21, -1.19]])
print("array : ")
# Create feature matrix with missing values in the categorical feature
X_with_nan = np.array([[np.nan, 0.87, 1.31], 
                       [np.nan, -0.67, -0.22]])
print("array with NaN")
print(X_with_nan)
# Train KNN learner
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:,1:], X[:,0])
# Predict missing values' class
imputed_values = trained_model.predict(X_with_nan[:,1:])
print("Imputed Values : ")
print(imputed_values)
print("After Reshape")
print(imputed_values.reshape(-1,1))
print("X with NaN : ")
print(X_with_nan[:,1:])

# Join column of predicted class with their other features
X_with_imputed = np.hstack((imputed_values.reshape(-1,1), X_with_nan[:,1:]))
print("X with imputed : ")
print(X_with_imputed)
# Join two feature matrices
fm=np.vstack((X_with_imputed, X))
print("Final Matrix : ")
print(fm)