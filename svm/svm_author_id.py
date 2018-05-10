#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from tools.email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import collections

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### create classifier
clf = SVC(kernel = 'rbf', C = 10000)

### cut size of training data set
#features_train = features_train[:int(len(features_train)/100)] 
#labels_train = labels_train[:int(len(labels_train)/100)] 

t0 = time() # time training
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t1 = time() # time predictions
pred = clf.predict(features_test)
print ("prediction time:", round(time()-t1, 3), "s")

preds = (10, 26, 50)
for p in preds:
    print('pred[' + str(p) + ']: ', pred[p])
    
counter = collections.Counter(pred)

print('Training Emails: ', counter[1] + counter[0] )        
print('Predicted Chris Emails: ', counter[1])

accuracy = accuracy_score(pred, labels_test)

print(accuracy)