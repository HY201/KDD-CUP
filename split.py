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

def getTwoTypesData(filename):
    raw = pd.read_csv(filename, sep='\t')
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
    
    for col in columns_numeric:
        arr = pd.cut(data[col], 10).astype('object')
        data[col] = arr
    
    data = labelEncoder(data, columns)    
    
#    return data[columns_discrete], data[columns_numeric]
    return data
    
def getNoMissingData(filename):
    data = pd.read_csv(filename, sep='\t')
    #data = raw.dropna(axis=0, how='all')
    #data = raw.dropna(axis=1, how='all')
    
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
        
    uniq_count = []
    for col in columns:
        uniq_count.append(len(data[col].unique()))
        
    #count_10 = 0
    #count_100 = 0
    #count_1000 = 0
    #count_10000 = 0
    #count_50000 = 0
    #for x in uniq_count:
    #    if x < 10:
    #        count_10 = count_10 + 1
    #    elif x < 100:
    #        count_100 = count_100 + 1
    #    elif x < 1000:
    #        count_1000 = count_1000 + 1
    #    elif x < 10000:
    #        count_10000 = count_10000 + 1
    #    else:
    #        count_50000 = count_50000 + 1
    #        
    #print "0-10:", count_10
    #print "10-100:", count_100
    #print "100-1000:", count_1000
    #print "1000-10000:", count_10000
    #print ">10000:", count_50000
    
    miss_count = []
    for col in columns:    
        count = 0
        for x in data[col].isnull():
            if x == True:
                count = count + 1
        miss_count.append(count)
    #print count_miss
    
    good_feature = []    
    for i in xrange(len(columns)):
        if miss_count[i] <1000 and uniq_count[i] > 2:
            good_feature.append(columns[i])
#    print good_feature

    data = data.dropna(axis=0, how='all')
    data = data.dropna(axis=1, how='all')    
    
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
    
    data.fillna(data.mean(), inplace=True)   
    
    for i in xrange(data.columns.shape[0]):
        col_name = data.columns.values[i]
        data[col_name].fillna(data[col_name].mode()[0], inplace=True)    
        
    data = labelEncoder(data, columns_discrete)
    
    
    return data[good_feature]
#    return data[columns_discrete], data[columns_numeric]