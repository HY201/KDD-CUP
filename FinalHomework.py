# -*- coding: utf-8 -*-
"""
Created on Thu May 19 08:53:47 2016

@author: admin
"""
from sklearn import tree
from sklearn import neighbors
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import MutiLabelEncoder as labelencoder
import random

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

#testing_label and test_data isn't used temperorily
def classify(traning_data, training_label, test_data, testing_label):
    fold = 2
    skf = StratifiedKFold(training_label, fold)
    roc_auc = 0    
    
    for train, test in skf:
        clf = BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5)    
    
        clf = clf.fit(traning_data.iloc[train], training_label.iloc[train])

        probas_  = clf.predict_proba(traning_data.iloc[test])
        fpr, tpr, thresholds = roc_curve(training_label.iloc[test], probas_[:, 0])
        roc_auc += auc(fpr, tpr)    
    
    return roc_auc / fold
    
def replaceMissingValue(data):
    data  = data.fillna(0)
    return data
    
def balance(data, label, ratio):
    data['label'] = label
    
    negative_data = data[data['label'] == -1]
    positive_data = data[data['label'] == 1]    
    
    negative_num = int(ratio * (label[label == -1]).shape[0])
    print negative_num
    
        
    for i in range(negative_num):
        ran = random.randint(0, negative_num - i)
        negative_data = negative_data.drop(negative_data.index[ran])
            
    data = negative_data.append(positive_data)    

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

#    balance(training_data, appe_label, 0.7)    
    
    #retrieve parts of data
    part_testing_data = testing_data.iloc[:, 220:223]
    part_training_data = training_data.iloc[:, 220:223]
    part_testing_data = balance(part_training_data, appe_label, 0.3)        
    part_training_data = balance(part_training_data, appe_label, 0.3)
    
    #transform caterical data into integer  
#    part_testing_data = labelEncoder(part_testing_data, ['Var221', 'Var222', 'Var223'])
#    part_training_data = labelEncoder(part_training_data, ['Var221', 'Var222', 'Var223'])
        
#    print part_testing_data
#    print "appe: %f" % classify(part_training_data, appe_label, part_testing_data, appe_label_test)    
#    print "churn: %f" % classify(part_training_data, churn_label, part_testing_data, churn_label_test)    
#    print "upsel: %f" % classify(part_training_data, upsel_label, part_testing_data, upsel_label_test)        
    
    
    
main()
    
    