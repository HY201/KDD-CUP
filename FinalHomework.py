# -*- coding: utf-8 -*-
"""
Created on Thu May 19 08:53:47 2016

@author: admin
"""
from sklearn import tree
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_classif
from sklearn.metrics import (precision_score, recall_score,f1_score, precision_recall_fscore_support)
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

import pandas as pd
import MutiLabelEncoder as labelencoder
import split as sp
import numpy as np
import random
import preprocessing

#data is divided into two parts : traning data and test data
def splitData(dataSet, num_of_testing):
    training_data = dataSet[0 : -num_of_testing]
    testing_data = dataSet[-num_of_testing : -1]
    return training_data, testing_data

def loadData(filePath):
#    df = pd.read_csv(filePath, sep='\t')
#    df = sp.getNoMissingData(filePath)
    df = preprocessing.getDataAfterPreprocessing(filePath)
#    df = sp.getTwoTypesData(filePath)
#    df = preprocessing.binningDiscreteVariables(df)
    training_data, testing_data = splitData(df, 1000)
    return training_data, testing_data

    
def loadLabel(filePath):
    df = pd.read_csv(filePath, header=None)
    training_label, testing_label = splitData(df, 1000)
    return training_label[0], testing_label[0]

def estimate(classfier, traning_data, training_label, test_data, testing_label):
    fold = 5
    skf = StratifiedKFold(training_label, fold)
    roc_auc = 0  
    f1_score_value = 0
    
    for train, test in skf:
        classfier = classfier.fit(traning_data.iloc[train], training_label.iloc[train])  
        
        #compute f1_score
        label_pred = classfier.predict(traning_data.iloc[test])
        
        #compute f1 score        
        roc_auc += roc_auc_score(training_label.iloc[test], label_pred)        
        f1_score_value += f1_score(training_label.iloc[test], label_pred, pos_label= 1)
        
    return roc_auc / fold, f1_score_value / fold  

def estimate_ensemble(classfier, traning_data, training_label, test_data, testing_label):
    fold = 5
    skf = StratifiedKFold(training_label, fold)
    roc_auc = 0  
    f1_score_value = 0
    
    
    for train, test in skf:
        classfier = classfier.fit(traning_data.iloc[train], training_label.iloc[train])  
        
        #compute f1_score
        label_pred = classfier.predict(traning_data.iloc[test])
        
        #compute f1 score        
        roc_auc += roc_auc_score(training_label.iloc[test], label_pred)        
        f1_score_value += f1_score(training_label.iloc[test], label_pred, pos_label= 1)
        
    return roc_auc / fold, f1_score_value / fold  

#testing_label and test_data isn't used temperorily
def classify(traning_data, training_label, test_data, testing_label):
    
#    print "decision tree :\n auc: %f f1_score: %f" % estimate(tree.DecisionTreeClassifier(max_depth=100, max_leaf_nodes= 100, class_weight={1:12}),traning_data, training_label, test_data, testing_label)       
#    print "adboost tree :\n auc: %f f1_score: %f" % estimate(AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=300, max_leaf_nodes= 300, class_weight={1:12}),
#                                                                                     algorithm="SAMME",
#                                                                                     n_estimators=100),traning_data, training_label, test_data, testing_label)        
#    print "gradientBoostingClassifier tree :\n auc: %f f1_score: %f" % estimate(GradientBoostingClassifier(max_depth = 100, max_leaf_nodes = 100,
#                                                                                     n_estimators=100),traning_data, training_label, test_data, testing_label)            
#    print "bagging tree :\n auc: %f f1_score: %f" % estimate(BaggingClassifier(tree.DecisionTreeClassifier(max_depth=300, max_leaf_nodes= 300, class_weight={1:12}),
#                                                                                     n_estimators=100),traning_data, training_label, test_data, testing_label)                                                                                                
    
    print "decision tree :\n auc: %f f1_score: %f" % estimate_ensemble(tree.DecisionTreeClassifier(max_depth=100, max_leaf_nodes= 100, class_weight={1:12}),traning_data, training_label, test_data, testing_label)           
    

def binning(data):
    data = data.values
    bins = np.linspace(0, 1, 10)
    digitized = np.digitize(data, bins)
    bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]

def selectModel(traning_data, training_label, test_data, testing_label):
    # Set the parameters by cross-validation
    tuned_parameters = [{'max_depth': [100, 200, 500], 'max_leaf_nodes': [100, 200, 500], 'class_weight' : [{1:12}]}]

    scores = ['roc_auc']

    for score in scores:
        clf = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=5, scoring=score)    
        
        clf.fit(traning_data, training_label)
#        print clf.best_params_

        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    
def replaceMissingValue(data):
    data  = data.fillna(0)
    return data
    
def balance(data, label, ratio):
    data['label'] = label
    
    negative_data = data[data['label'] == -1]
    positive_data = data[data['label'] == 1]   
    
    negative_num = int(ratio * (label[label == -1]).shape[0])

    random_drop = random.sample(range(label[label == -1].shape[0]), negative_num)
    negative_data = negative_data.reset_index(drop=True).drop(random_drop)
    
    data = negative_data.append(positive_data) 
    label = data['label']
    
    del data['label']

    return data, label
    
def selectFeature(data, label):
    clf = GradientBoostingClassifier(learning_rate=0.02, n_estimators=100, max_leaf_nodes=6)
    clf.fit(data, label)
    model = SelectFromModel(clf, threshold='0.1*mean', prefit=True)
    data = model.transform(data)
    return data
#        clf = ExtraTreesClassifier()
##    clf = SVR(kernel="linear")
#    clf = clf.fit(data, label)
##linearsvc    
##    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data, label)
##VarianceThreshold
#    sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
#    sel.fit_transform(data)
##bagging
##    bagging = BaggingClassifier(tree.DecisionTreeClassifier(max_depth=500, max_leaf_nodes= 500) ,max_samples=0.5, max_features=0.5)
##    bagging = bagging.fit(data,label)
##Random forest
##    randomfor = RandomForestClassifier(n_estimators=10)
##    randomfor = randomfor.fit(data, label)
##decisionTree
##    decisiontree = tree.DecisionTreeClassifier(max_depth=500, max_leaf_nodes= 500)
##    decisiontree.fit(data,label)
##RFE
#    selector1 = RFE(clf, 60, step=1)
#    selector1 = selector1.fit(data, label)
##frommodel
#    selector2 = SelectFromModel(clf, prefit=True)
##    selector3 = SelectFromModel(decisiontree, prefit=True)
##    selector4 = SelectFromModel(bagging, prefit=True)
##KBest
##    selector5 = SelectKBest(chi2, k=100).fit(data, label)
##    print selector1.transform(data)
##merge features
#    selected_data = np.append(selector1.transform(data).T,selector2.transform(data).T,axis = 0)
#    selected_data = np.array(list(set([tuple(x) for x in selected_data])))
#    return selected_data.T
##    return selector2.transform(data)

#e.g. columns = ['Var1', 'Var2']
def labelEncoder(dataframe, columns):
    return labelencoder.MultiColumnLabelEncoder(columns=columns).transform(dataframe)

def labelEncoder2(dataframe):
    return labelencoder.MultiColumnLabelEncoder().transform(dataframe)
    
def main():
    #read data from files
#    training_data, testing_data = loadData('step_preprocess.data')
#    print 'start to load and preprocess data'
    training_data, testing_data = loadData('orange_small_train.data')
    
#    training_data, testing_data = loadData('new.data')
    appe_label, appe_label_test = loadLabel('orange_small_train_appetency.labels')
    churn_label, churn_label_test = loadLabel('orange_small_train_churn.labels')
    upsel_label, upsel_label_test = loadLabel('orange_small_train_upselling.labels')
    
    #fill missing value 
#    training_data = replaceMissingValue(training_data)
#    testing_data = replaceMissingValue(testing_data)        
    
    
    print 'feature selection'
#    part_testing_data_appe = pd.DataFrame(selectFeature(testing_data, appe_label_test))
#    part_training_data_appe = pd.DataFrame(selectFeature(training_data, appe_label))
    part_testing_data_churn = pd.DataFrame(selectFeature(testing_data, churn_label_test))
    part_training_data_churn = pd.DataFrame(selectFeature(training_data, churn_label))
#    part_testing_data_upsel = pd.DataFrame(selectFeature(testing_data, upsel_label_test))
#    part_training_data_upsel = pd.DataFrame(selectFeature(training_data, upsel_label))

#    part_testing_data = testing_data
#    part_training_data = training_data

#    print part_training_data
#    print 'balance data'    
    
#    part_training_data_appe, appe_label = balance(part_training_data_appe, appe_label, 0)
#    part_training_data_churn, churn_label = balance(part_testing_data_churn, churn_label, 0)
#    part_training_data_upsel, upsel_label = balance(part_testing_data_upsel, upsel_label, 0)    
    
#    part_testing_data = testing_data.iloc[:, 0:50]
#    part_training_data = training_data.iloc[:, 0:50]
#    part_testing_data = balance(part_training_data, appe_label, 0.3)        
#    part_training_data = balance(part_training_data, appe_label, 0.3)   

#    print 'start to select model'
#    selectModel(part_training_data_churn, churn_label, part_testing_data_churn, churn_label_test)    
     
#    print part_testing_data
#    print 'star to classify'
    
#    print churn_label.values
    print 'churn:'
    classify(part_training_data_churn, churn_label, part_testing_data_churn, churn_label_test)    
    
#    print "\nappe:"  
#    classify(part_training_data_appe, appe_label, part_testing_data_appe, appe_label_test)    
#
#    print "\nupsel:" 
#    classify(part_training_data_upsel, upsel_label, part_testing_data_upsel, upsel_label_test)        

#    print 'star to classify'
#    print "appe: %f" % classify(training_data, appe_label, testing_data, appe_label_test)    
#    print "churn: %f" % classify(training_data, churn_label, testing_data, churn_label_test)    
#    print "upsel: %f" % classify(training_data, upsel_label, testing_data, upsel_label_test)              
    
def loadData2(filePath):
    discret, numeric = sp.getNoMissingData(filePath)
    training_data, testing_data = splitData(discret, 1000)
    training_data2, testing_data2 = splitData(numeric, 1000)    
    
    return training_data, training_data2

def loadLabel2(filePath):
    df = pd.read_csv(filePath, header=None)
    training_label, testing_label = splitData(df, 1000)
    return training_label

def classify2(dis_data, numeric_data, t_label):
    fold = 5
    skf = StratifiedKFold(t_label, fold)
    roc_auc = 0  
    f1_score_value = 0

    clf1 = LogisticRegression()
    clf2 = GradientBoostingClassifier()
#    clf3 = tree.DecisionTreeClassifier(max_depth=500, max_leaf_nodes= 500, class_weight={1:12})
    clf3 = GradientBoostingClassifier()
    
    for train, test in skf:
        clf3 = clf3.fit(dis_data.iloc[train], t_label.iloc[train])
        
        #compute auc
        probas_  = clf3.predict_proba(dis_data.iloc[test])
        fpr, tpr, thresholds = roc_curve(t_label.iloc[test], probas_[:, 0])
        roc_auc += auc(fpr, tpr)    
        
        #compute f1_score
        label_pred = clf3.predict(dis_data.iloc[test])
        
        f1_score_value += f1_score(t_label.iloc[test], label_pred, pos_label= 1)
        
    return roc_auc / fold, f1_score_value / fold     

def main2():
    print 'start to load and preprocess data'
    discret_data, numeric_data= loadData2('orange_small_train.data')
    
#    training_data, testing_data = loadData('new.data')
    appe_label = loadLabel2('orange_small_train_appetency.labels')
#    churn_label = loadLabel2('orange_small_train_churn.labels')
#    upsel_label = loadLabel2('orange_small_train_upselling.labels')    
    
    discret_data = pd.DataFrame(selectFeature(discret_data, appe_label))
    numeric_data = pd.DataFrame(selectFeature(numeric_data, appe_label))
    
#    print discret_data.shape
#    print numeric_data.shape
#    print appe_label.values
#    print appe_label.iloc[:, 0]
    
    print "auc: %f f1_score: %f" % classify2( numeric_data, discret_data, appe_label.iloc[:, 0])
    
    
main()

    
    