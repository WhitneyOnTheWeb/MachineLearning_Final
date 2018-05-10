#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import collections
import pandas as pd
import numpy as np
import sys
sys.path.append( "C:/JupyterNotebook/Machine_Learning/ud120-projects/" )
from tools.feature_format import featureFormat, targetFeatureSplit
from time import time
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif


data_dict = pickle.load(open("final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = 'tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)


#Classifier
clf = tree.DecisionTreeClassifier()

t0 = time() # time training
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t1 = time() # time predictions
pred = clf.predict(features_test)
print ("prediction time:", round(time()-t1, 3), "s")

accuracy = accuracy_score(pred, labels_test)

#Number of True Positives
tps = [ (x, y) for x, y in zip(pred, labels_test) if x == y and x == 1.0]

#Precision and Recall Scores
ps = precision_score(pred, labels_test)
rs = recall_score(pred, labels_test)

print('Accuracy:', accuracy)
print('Precision:', ps)
print('Recall:', rs)
print('# POIs:', int(sum(pred)))
print('# Test Features', len(pred))
print('TPs:', len(tps))

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

tps = precision_score(predictions, true_labels)
trs = recall_score(predictions, true_labels)

print(tps, trs)