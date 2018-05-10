#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from tools.feature_format import featureFormat, targetFeatureSplit
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import collections
from time import time
import pandas as pd




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../ud120-projects/final_project/final_project_dataset.pkl", "rb") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = 'total_payments'
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list, remove_any_zeroes = True )
poi, finance_features = targetFeatureSplit( data )
d = pd.DataFrame(data_dict)

stocks = [] 
for key, value in data_dict.items():
    if value['exercised_stock_options'] != 'NaN':
        stocks.append(value['exercised_stock_options'])

print('Max stock: ', d.loc['exercised_stock_options'].idxmax(), max(stocks))
print('Min stock: ', d.loc['exercised_stock_options'].idxmin(), min(stocks))

salary = [] 
for key, value in data_dict.items():
    if value['salary'] != 'NaN':
        salary.append(value['salary'])

print('Max salary: ', d.loc['salary'].idxmax(), max(salary))
print('Min salary: ', d.loc['salary'].idxmin(), min(salary))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
pred_stock = numpy.array( [[float(min(stocks))], [1000000.], [float(max(stocks))]] )
pred_salary = numpy.array( [[float(min(salary))], [200000.], [float(max(salary))]] )

print('Scaled stock: ', scaler.fit_transform(pred_stock))
print('Scaled salary: ', scaler.fit_transform(pred_salary))

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

### create classifier
clf = KMeans(n_clusters = 2)

t0 = time() # time training
clf.fit(finance_features)
print ("training time:", round(time()-t0, 3), "s")

t1 = time() # time predictions
pred = clf.predict(finance_features)
print ("prediction time:", round(time()-t1, 3), "s")

accuracy = accuracy_score(pred, poi)

print(accuracy)

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("no predictions object named pred found, no clusters to plot")
