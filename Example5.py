import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv('iris.csv')
print("Whole data : ")
print(df)
if df.shape[1] > 4:
    df_with_nan = df[df.iloc[:, 4].isnull()]
else:
    print("The dataset does not contain 5 columns as expected.")
print("Data with NaN values : ")
print(df_with_nan)
df_no_nan=df.dropna()
print("Data without NaN values : ")
print(df_no_nan)
x=df_no_nan.iloc[:,:-1].values
y=df_no_nan.iloc[:,-1].values
clf=KNeighborsClassifier(5,weights='distance')
model=clf.fit(x, y)
imputedVal=model.predict(df_with_nan.iloc[:,:-1].values)
imputedVal.reshape(-1,1)
print("Imputed Values : ")
print(imputedVal)
df_nan_imputed=np.hstack((df_with_nan.iloc[:,:-1].values,imputedVal.reshape(-1,1)))
print("\nImputed values hstack with dataset with Nan")
print(df_nan_imputed)
finalDf=pd.concat([pd.DataFrame(df_nan_imputed),df_no_nan])
print("\n Final Dataset : ")
print(finalDf)
pd.DataFrame(finalDf).to_csv("iris_processed.csv",header=False,index=False)
