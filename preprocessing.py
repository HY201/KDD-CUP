# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:00:56 2016

@author: asus1
"""

import numpy as np
import pandas as pd
import MutiLabelEncoder as labelencoder

def labelEncoder(dataframe, columns):
    return labelencoder.MultiColumnLabelEncoder(columns=columns).transform(dataframe)

def loadData(filename):
    data = pd.read_csv(filename, sep='\t')
    return data    

def getColumns(data):
    columns =[]
    for i in range(data.columns.shape[0]):
        col_name = data.columns.values[i]
        columns.append(col_name)
    return columns
    
def getDiscreteColumns(data):
    columns_discrete = []
    for i in range(data.columns.shape[0]):
        col_name = data.columns.values[i]
        if data[col_name].dtypes == np.object or data[col_name].dtypes == np.bool:
            columns_discrete.append(col_name)
    return columns_discrete
            
def getNumericColumns(data):
    columns_numeric = []
    for i in range(data.columns.shape[0]):
        col_name = data.columns.values[i]
        if data[col_name].dtypes != np.object and data[col_name].dtypes != np.bool:
            columns_numeric.append(col_name)
    return columns_numeric
    
def getUniqueCount(data):
    columns = getColumns(data)    
    uniq_count = []
    for col in columns:
        uniq_count.append(len(data[col].unique()))
    return uniq_count

def getMissingCount(data):
    columns = getColumns(data)
    miss_count = []
    for col in columns:    
        count = 0
        for x in data[col].isnull():
            if x == True:
                count = count + 1
        miss_count.append(count)
    return miss_count

def addDiscription(data):
    columns = getColumns(data)
    
    addition_columns = []    
    for i in range(231, 461):
        col_name = 'Var' + str(i)
        addition_columns.append(col_name)
        
    col_index = 0
    for col in columns:
        newCol = data[col].isnull()
        data[addition_columns[col_index]] = newCol
        col_index = col_index + 1
    
    return data

def binaryDiscrete(data):
    columns_discrete = getDiscreteColumns(data)
    data = labelEncoder(data, columns_discrete)
    return data

def binningDiscreteVariables(data):
    print 'binningDiscreteVariables'
    discreteColumns = getDiscreteColumns(data)
#    i=0
    for column in discreteColumns:
#        i=i+1
#        print i
        if len(data[column].unique()) > 1000:
            values = data[column].unique()
            counts={}
            for value in data[column]:
                if value in counts:
                    counts[value]=counts[value]+1
                else:
                    counts[value]=1
            for value in values:
#                counts = [x for x in data[column] if x == value]
#                count = len(counts)
                if(counts[value] < len(data[column])/5000):
                    data[column] = data[column].replace(value,column+'_another1')
                elif(counts[value] < len(data[column])/1000):
                    data[column] = data[column].replace(value,column+'_another2')
                elif(counts[value] < len(data[column])/500):
                    data[column] = data[column].replace(value,column+'_another3')
#                elif(counts[value] < len(data[column])/300):
#                    data[column] = data[column].replace(value,column+'_another4')
#    print data
    return data

def imputeMissing(data):
    columns_numeric = getNumericColumns(data)    
    for col in columns_numeric:
        data[col].fillna(-1, inplace=True)
    return data

def dropNan(data):
    data = data.dropna(axis=1, how='all')
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
    return data

def getDataAfterPreprocessing(filename):
#    filename = 'orange_small_train.data'
    data = loadData(filename)
#    data = addDiscription(data)  
    data = imputeMissing(data)
    data = binningDiscreteVariables(data)
    data = binaryDiscrete(data)
    data = dropNan(data)    
    return data
    
    
#print getDataAfterPreprocessing()

#
#good_feature = []    
#for i in xrange(len(columns)):
#    if miss_count[i] < 50000*0.02 and uniq_count[i] > 2:
#        good_feature.append(columns[i])
#print len(good_feature)
