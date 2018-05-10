#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)
from time import time
from sklearn import tree
import collections
from sklearn.metrics import accuracy_score
import sys
sys.path.append( "C:/JupyterNotebook/Machine_Learning/ud120-projects/" )
### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "your_word_data.pkl" 
authors_file = "your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here

clf = tree.DecisionTreeClassifier(min_samples_split = 40)
t0 = time() # time training
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t1 = time() # time predictions
pred = clf.predict(features_test)
print ("prediction time:", round(time()-t1, 3), "s")

accuracy = accuracy_score(pred, labels_test)

print(accuracy)
print(len(features_train))

fi =  clf.feature_importances_.tolist()
imp_fi = []

for f in fi:
    if f > .2:
        imp_fi.append([f , fi.index(f)])

print(imp_fi)

fn = vectorizer.get_feature_names()
print(fn[18849], fn[21323])