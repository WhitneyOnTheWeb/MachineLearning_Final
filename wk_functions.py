import sys
sys.path.append("../ud120-projects/tools/")
import pickle
import collections
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from time import time
import pprint
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

def line():
    print('-'*80)

def check_for_outliers_95(f1, f2, data_dict):
    f_list = []
    f_95 = []
    f1_95 = []
    f2_95 = []

    # Get list of persons with values from the two specified financial features
    for key, value in data_dict.items():
        f_list.append([key, value[f1], value[f2]])

    # Get rid of any entries that have NaN for one or both features, since these are not outliers
    for i in f_list:
        if i[1] != 'NaN' and i[2] != 'NaN':
            f_95.append(i)

    # Test if values for both features fall into the 95th percentile, and if they do, print
    #the name of the person
    for i in f_95:
        f1_95.append(float(i[1]))
        f2_95.append(float(i[2]))
    f1_95 = float(np.percentile(f1_95, 95))
    f2_95 = float(np.percentile(f2_95, 95))

    for key, value in data_dict.items():
        if value[f1] != 'NaN' and value[f2] != 'NaN':
            if float(value[f1]) >= f1_95 and float(value[f2]) >= f2_95:
                print(key, value[f1], value[f2])

def check_for_outliers_name(data_dict):
    #Print names of employees out into 4 columns, reference: 
    # https://stackoverflow.com/questions/9989334/create-nice-column-output-in-python
    s = []
    for person in data_dict.keys():
        s.append(person)
        if len(s) == 4:
            print ('{:<30}{:<30}{:<30}{:<30}'.format(s[0],s[1],s[2],s[3]))
            s = []


def check_for_outliers_NaNs(data_dict, features_list):
    all_nans_list = []
    for key, value in data_dict.items():
        NaNs = 0
        for j, feature in enumerate(features_list):
            if value[feature] == 'NaN':
                NaNs += 1
            if NaNs == len(features_list) - 1:
                all_nans_list.append(key)
    print('Employees containing all NaNs:')
    print(all_nans_list)
        

def xval_classifier(clf, features, labels, cv):
    # Compare classifiers using cross validation
    tn = 0
    fn = 0
    tp = 0
    fp = 0

    for train_index, test_index in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []

        for i in train_index:
            features_train.append(features[i])
            labels_train.append(labels[i])
        for j in test_index:
            features_test.append(features[j])
            labels_test.append(labels[j])
        clf.fit(features_train, labels_train)
        preds = clf.predict(features_test)

        for pred, actual in zip(preds, labels_test):
            if pred == 0 and actual == 0:
                tn += 1
            elif pred == 0 and actual == 1:
                fn += 1
            elif pred == 1 and actual == 0:
                fp += 1
            elif pred == 1 and actual == 1:
                tp += 1

        pred_count = tn + fn + fp + tp
        #print(pred_count)

        if tp > 0:
            acc = round(np.float64(tp + tn) / pred_count, 2)
            prec = round(np.float64(tp) / (tp + fp), 2)
            re = round(np.float64(tp) / (tp + fn), 2)

    return acc, prec, re


def best_features(max_feature, features, labels, features_list):
    sel = SelectKBest(f_classif, k = max_feature)
    sel.fit(features, labels)
    stop = np.sort(sel.scores_)[::-1][max_feature]
    selected_features_list = [f for i, f in enumerate(features_list[1:]) if sel.scores_[i] >= stop]
    selected_features_list = ['poi'] + selected_features_list 
    selected_features = sel.fit_transform(features, labels)
    return selected_features_list, selected_features


def tune_RandomForest(rf_tuning_parameters, sss, bf_rf_recall, labels):
    line()
    t = time()
    print('Beginning tuning RandomForest...')
    print('Please wait...')
    RF = GridSearchCV(RandomForestClassifier(), rf_tuning_parameters, cv = sss, scoring = 'recall')
    RF.fit(bf_rf_recall, labels)
    print('Best Parameters:')
    print(RF.best_params_)
    print('Time Spent Tuning RF: {0}s'.format(round(time() - t, 2)))

    rf_clf = RF.best_estimator_
    print('Tuned RandomForest Metrics:')
    return rf_clf


def tune_AdaBoost(ada_tuning_parameters, sss, bf_ada_recall, labels):
    t = time()
    print('Beginning tuning AdaBoost...')
    print('Please wait...')
    ADA = GridSearchCV(AdaBoostClassifier(), ada_tuning_parameters, cv = sss, scoring = 'recall')
    ADA.fit(bf_ada_recall, labels)
    print('Best Parameters:')
    print(ADA.best_params_)
    print('Time Spent Tuning AdaBoost: {0}s'.format(round(time() - t, 2)))

    ada_clf = ADA.best_estimator_
    print('Tuned AdaBoost Metrics:')
    return ada_clf