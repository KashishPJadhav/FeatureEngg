import pandas as pd
import numpy as np
 
#Create a DataFrame
d ={
    'Name':['Alisa','Bobby','jodha','jack','raghu','Cathrine',
            'Alisa','Bobby','kumar','Alisa','Alex','Cathrine'],
    'Age':[26,24,23,22,23,24,26,24,22,23,24,24],
      
    'Score':[85,63,55,74,31,77,85,63,42,62,89,77]}
df=pd.DataFrame(d)
print("Original Data Frame : ")
print(df)

dupe=df.duplicated()
print("\nIs Duplicated : ")
print(dupe)

df1=df.drop_duplicates()
print("\nDropping Duplicates : ")
print(df1)

df_keeplast=df.drop_duplicates(keep='last')
print("\nDropping Duplicates keeping last : ")
print(df_keeplast)

df_keepfirst=df.drop_duplicates(keep='first')
print("\nDropping Duplicates keeping first : ")
print(df_keepfirst)

df_column=df.drop_duplicates(['Name'],keep='last')
print("\nDropping Duplicates by column name 'Name' keeping last : ")
print(df_column)
