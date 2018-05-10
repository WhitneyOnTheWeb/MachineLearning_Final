#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
import sklearn.ensemble
from sklearn.ensemble import AdaBoostClassifier
import collections
from sklearn.metrics import accuracy_score
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
#grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
#bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
#grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
#bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
#plt.xlim(0.0, 1.0)
#plt.ylim(0.0, 1.0)
#plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
#plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
#plt.legend()
#plt.xlabel("bumpiness")
#plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

clf = AdaBoostClassifier()

### cut size of training data set
#features_train = features_train[:int(len(features_train)/100)] 
#labels_train = labels_train[:int(len(labels_train)/100)] 

t0 = time() # time training
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t1 = time() # time predictions
pred = clf.predict(features_test)
print ("prediction time:", round(time()-t1, 3), "s")

preds = (10, 45, 50)
for p in preds:
    print('pred[' + str(p) + ']: ', pred[p])
    
counter = collections.Counter(pred)

print('Total Features: ', counter[1] + counter[0] )        
print('Type 1: ', counter[1])
print('Type 0: ', counter[0])

accuracy = accuracy_score(pred, labels_test)

print(accuracy)


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
