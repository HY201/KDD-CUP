# -*- coding: utf-8 -*-
"""
Created on Thu May 19 20:34:56 2016

@author: asus1
"""
import numpy as np
import pandas as pd
import MutiLabelEncoder as labelencoder

def labelEncoder(dataframe, columns):
    return labelencoder.MultiColumnLabelEncoder(columns=columns).transform(dataframe)

raw = pd.read_csv('orange_small_train.data', sep='\t')
data = raw.dropna(axis=0, how='all')
data = raw.dropna(axis=1, how='all')

data.fillna(data.mean(), inplace=True)   

for i in range(data.columns.shape[0]):
    col_name = data.columns.values[i]
    data[col_name].fillna(data[col_name].mode()[0], inplace=True)

newData = data
for i in range(data.columns.shape[0]):
    col_name = data.columns.values[i]
    temp = data.loc[0, col_name]
    flag = 0    
    for x in data[col_name]:
        if x != temp:
            flag = 1
    if flag == 0:
        newData = newData.drop(col_name, axis=1)
data = newData

#print data

columns = []
columns_discrete = []
columns_numeric = []
for i in range(data.columns.shape[0]):
    col_name = data.columns.values[i]
    columns.append(col_name)
    if data[col_name].dtypes == np.object:
        columns_discrete.append(col_name)
    else:
        columns_numeric.append(col_name)
        
#print columns_discrete,
#print columns_numeric

data = labelEncoder(data, columns_discrete)

for col in columns:
    arr = pd.cut(data[col], 10).astype('object')
    data[col] = arr

data = labelEncoder(data, columns)    
    
#print data
        
data.to_csv('new.data', sep='\t', index=False)
print 'complete'