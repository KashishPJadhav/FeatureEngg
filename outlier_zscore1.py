import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

df=pd.read_csv('diabetes2.csv',header=None);
df.describe()

x=df.iloc[:,0:1].values

plt.hist(x,density=False,bins=30)

def outlier_zscore(ys):
    thresh=3
    
    mean=np.mean(ys)
    stde=np.std(ys)
    z_scores=[(y-mean)/stde for y in ys]
    return np.where(np.abs(z_scores)>thresh)

y1=outlier_zscore(x)
print("outliers using def : ")
print(y1)

print("Z score od x : ")
zs=zscore(x)
print(zs)

y2=np.where(np.abs(zs)>3)
print("Outliers usind directly 'zscore' : ")
print(y2)

def outlier_iqr(ys):
    q1,q2,q3=np.percentile(ys,[25,50,75])
    iqr=q3-q1
    lower_bound=q1-(iqr*1.5)
    upper_bound=q3+(iqr*1.5)
    outl_indices=np.where((ys>upper_bound) | (ys<lower_bound))
    return q1,q2,q3,outl_indices
  
y3=outlier_iqr(x)
print("Outliers using 3 quartiles : ")
print(y3)

arr = [4,7,8,9,10,12,12,14,15,18,20,75]
print("Array : ",arr)
quartile_1,quartile_2, quartile_3 = np.percentile(arr, [25,50,75])
print("Quartiles of this array : ")
print(quartile_1,quartile_2,quartile_3)
y2 = outlier_iqr(arr)
print("Outlier in the array : ")
print(y2)