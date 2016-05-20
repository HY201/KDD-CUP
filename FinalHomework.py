# -*- coding: utf-8 -*-
"""
Created on Thu May 19 08:53:47 2016

@author: admin
"""
from sklearn import tree
from sklearn import neighbors
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
import pandas as pd
import MutiLabelEncoder as labelencoder

#data is divided into two parts : traning data and test data
def splitData(dataSet, num_of_testing):
    training_data = dataSet[0 : -num_of_testing]
    testing_data = dataSet[-num_of_testing : -1]
    return training_data, testing_data

def loadData(filePath):
    df = pd.read_csv(filePath, sep='\t')
    training_data, testing_data = splitData(df, 1000)
    return training_data, testing_data
    
def loadLabel(filePath):
    df = pd.read_csv(filePath, header=None)
    training_label, testing_label = splitData(df, 1000)
    return training_label[0], testing_label[0]

def classify(traning_data, training_label, test_data, testing_label):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(traning_data, training_label)
    
#    test_label = clf.predict(test_data)
#    return test_label
    probas_  = clf.predict_proba(test_data)
    fpr, tpr, thresholds = roc_curve(testing_label, probas_[:, 0])
    roc_auc = auc(fpr, tpr)    
    
    return roc_auc

def classify2(traning_data, training_label, test_data, testing_label):
    clf = neighbors.KNeighborsClassifier(15)
    clf = clf.fit(traning_data, training_label)

#    test_label = clf.predict(test_data)
    probas_  = clf.predict_proba(test_data)
    fpr, tpr, thresholds = roc_curve(testing_label, probas_[:, 0])
    roc_auc = auc(fpr, tpr)    
    
    return roc_auc

def classify3(traning_data, training_label, test_data, testing_label):
    clf = BaggingRegressor(DecisionTreeRegressor())
    clf = clf.fit(traning_data, training_label)

#    test_label = clf.predict(test_data)
#    probas_  = clf.predict_proba(test_data)
    probas_ = clf.predict(test_data)
#    fpr, tpr, thresholds = roc_curve(testing_label, probas_[:, 0])
#    roc_auc = auc(fpr, tpr)    
    
    return probas_
    
def replaceMissingValue(data):
    data  = data.fillna(0)
    return data

#e.g. columns = ['Var1', 'Var2']
def labelEncoder(dataframe, columns):
    return labelencoder.MultiColumnLabelEncoder(columns=columns).transform(dataframe)
    
def main():
    #read data from files
    training_data, testing_data = loadData('orange_small_train.data')
    appe_label, appe_label_test = loadLabel('orange_small_train_appetency.labels')
    churn_label, churn_label_test = loadLabel('orange_small_train_churn.labels')
    upsel_label, upsel_label_test = loadLabel('orange_small_train_upselling.labels')
    
    #fill missing value 
    training_data = replaceMissingValue(training_data)
    testing_data = replaceMissingValue(testing_data)    

    #retrieve parts of data
    part_testing_data = testing_data.iloc[:, 220:223]
    part_training_data = training_data.iloc[:, 220:223]
    
    #transform caterical data into integer  
    part_testing_data = labelEncoder(part_testing_data, ['Var221', 'Var222', 'Var223'])
    part_training_data = labelEncoder(part_training_data, ['Var221', 'Var222', 'Var223'])
        
    print part_testing_data
    print classify(part_training_data, appe_label, part_testing_data, appe_label_test)    
    
#    print training_data    
#    print appe_label.astype(str).values
    
#    part_training_data = training_data.iloc[:, 5:7]
#    part_testing_data = testing_data.iloc[:, 5:7]
#    print classify(part_training_data, appe_label, part_testing_data, appe_label_test)    
#    print classify(training_data, appe_label, testing_data, appe_label_test)    
    
    
main()
    
    