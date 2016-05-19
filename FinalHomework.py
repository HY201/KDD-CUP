# -*- coding: utf-8 -*-
"""
Created on Thu May 19 08:53:47 2016

@author: admin
"""
from sklearn import tree
from sklearn import neighbors
from sklearn.metrics import roc_curve, auc
import pandas as pd

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

def classify(traning_data, training_label, test_data):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(traning_data, training_label)

    test_label = clf.predict(test_data)
    return test_label

def classify2(traning_data, training_label, test_data, testing_label):
    clf = neighbors.KNeighborsClassifier(15)
    clf = clf.fit(traning_data, training_label)

#    test_label = clf.predict(test_data)
    probas_  = clf.predict_proba(test_data)
    fpr, tpr, thresholds = roc_curve(testing_label, probas_[:, 0])
    roc_auc = auc(fpr, tpr)    
    
    return roc_auc
    
def replaceMissingValue(data):
    data  = data.fillna(0)
    return data
    
def main():
    training_data, testing_data = loadData('orange_small_train.data')
    appe_label, appe_label_test = loadLabel('orange_small_train_appetency.labels')
    churn_label, churn_label_test = loadLabel('orange_small_train_churn.labels')
    upsel_label, upsel_label_test = loadLabel('orange_small_train_upselling.labels')
    
    training_data = replaceMissingValue(training_data)
    testing_data = replaceMissingValue(testing_data)    

#    part_training_data = training_data.iloc[:, 220:223]
#    part_testing_data = testing_data.iloc[:, 220:223]
#    print classify(part_training_data, appe_label.astype(str), part_testing_data)    
    
#    print training_data    
#    print appe_label.astype(str).values
    
    part_training_data = training_data.iloc[:, 5:7]
    part_testing_data = testing_data.iloc[:, 5:7]
    print classify2(part_training_data, appe_label, part_testing_data, appe_label_test)    
    
    
main()
    
    