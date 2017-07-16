#!/usr/bin/python

import sys
sys.path.append("../ud120-projects/tools/")
import pickle
import collections
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from time import time
import pprint
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from wk_functions import check_for_outliers_95, check_for_outliers_name, check_for_outliers_NaNs, xval_classifier, best_features, tune_RandomForest, tune_AdaBoost, line
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

import warnings
warnings.filterwarnings("ignore")

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

##Shortened list of features to important ones based on findings from previously running the script
poi_label = ['poi']
#financial_features = ['salary', 'total_payments', 'loan_advances', 'bonus',
#                      'deferred_income', 'total_stock_value',
#                      'expenses', 'exercised_stock_options', 'long_term_incentive',
#                      'restricted_stock']
                      # Removed other
#email_features = ['from_poi_to_this_person', 'shared_receipt_with_poi'] 
                  # Removed email_address
				  
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                      'expenses', 'exercised_stock_options', 'long_term_incentive',
                      'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'shared_receipt_with_poi']

#financial_features = ['salary', 'deferral_payments']
#email_features = ['to_messages', 'from_poi_to_this_person']

line()
features_list = poi_label + financial_features + email_features
print('Total Starting Features:', len(features_list))

### Load the dictionary containing the dataset
with open("final_project/final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
    #data_dict = pd.DataFrame(data_dict)
print('Total Data Points:', len(data_dict))

NaNs = [0 for i in range(len(features_list))]
for i, person in enumerate(data_dict.values()):
    for j, feature in enumerate(features_list):
        if person[feature] == 'NaN':
            NaNs[j] += 1
line() # Check for NaNs in each feature
print('# NaNs in Each Feature:')
for i, feature in enumerate(features_list):
    print(feature + ': ', NaNs[i])

### Task 2: Remove outliers
line()
print('95% Percentile Financial Feature Values')
line()
print('salary/bonus:')
check_for_outliers_95('bonus', 'salary', data_dict)
    #   LAY KENNETH L 7000000 1072321
    #   SKILLING JEFFREY K 5600000 1111258
    #   TOTAL 97343619 26704229
line()
print('exercised stock options/long term incentive:')
check_for_outliers_95('exercised_stock_options', 'long_term_incentive', data_dict)
    #   LAY KENNETH L 34348384 3600000
    #   TOTAL 311764000 48521928
line()
print('TOTAL is an outlier that sums the rest of the data')
print('It should be removed from the dataset')
data_dict.pop('TOTAL')

line()
print('Check employee names:')
check_for_outliers_name(data_dict)
line()
print('THE TRAVEL AGENCY IN THE PARK is not an employee, and should be removed from the list')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

line()
check_for_outliers_NaNs(data_dict, features_list)
print('LOCKHART EUGENE E has only NaN values, so he should be removed from the list')
data_dict.pop('LOCKHART EUGENE E')

line()
print('Updated Total Data Points:', len(data_dict))
line()

# Check number of persons of interest
npoi = 0
for p in data_dict.values():
    if p['poi']:
        npoi += 1
print ("Total POIs: ", npoi)
print ("Total Non-POIs: ", len(data_dict) - npoi)
line()

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

#### Comment out below section to score classifiers without the new features
## Create new features: to_poi_ratio and from_poi_ratio
for person in my_dataset.values():
    person['to_poi_ratio'] = 0
    person['from_poi_ratio'] = 0
    person['blank'] = 0
    if float(person['from_messages']) > 0:
        person['to_poi_ratio'] = \
            float(person['from_this_person_to_poi']) / float(person['from_messages'])
    if float(person['from_messages']) > 0:
        person['from_poi_ratio'] = \
            float(person['from_poi_to_this_person']) / float(person['to_messages'])

features_list.extend(['to_poi_ratio', 'from_poi_ratio', 'blank'])
print(features_list)
print('New features created: to_poi_ratio, from_poi_ratio')
print('These features calculate the ratio of emails to or from POIs out')
print('of total sent or received')
line()
print('to_poi_ratio: from_this_person_to_poi / from_messages')
print('from_poi_ratio: from_poi_to_this_person / to_messages')
print('Features list was updated with to_poi_ratio')
print('Updated Total Features:', len(features_list) - 1)
line()
#### End comment section

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

## Use StratifiedShuffleSplit() to try to identify best features for the model
sss = StratifiedShuffleSplit(labels, 1000, random_state = 666)
print(sss)
print('Split Data Set Using StratifiedShuffleSplit for best feature identification')

line()    

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.

# Assign variables for classifier comparisons, these were taken from the suggestions in
# the choose_your_own algorithm project: AdaBoost and Random Forest

print('Begin training AdaBoost and RandomForest Classifiers...')
print('Please wait...')
print('Training {0} Features...'.format(len(features_list) - 1))
line()

scores = []
ada_accuracy = []	
ada_precision = []
ada_recall = []
rf_accuracy = []
rf_precision = []
rf_recall = []

for i in range(len(features[0])):
    t = time()
    sel = SelectKBest(f_classif, k = i + 1)
    sel.fit(features, labels)
    reduced_features = sel.fit_transform(features, labels)
    stop = np.sort(sel.scores_)[::-1][i]
    selected_features_list = [f for j, f in enumerate(features_list[1:]) if sel.scores_[j] >= stop]
    selected_features_list = ['poi'] + selected_features_list
    
    # Run metrics on AdaBoost Classifier
    ada = AdaBoostClassifier(random_state = 666)
    acc, prec, re = xval_classifier(ada, reduced_features, labels, sss)
    ada_accuracy.append(round(float(acc), 2))
    ada_precision.append(round(float(prec), 2))
    ada_recall.append(round(float(re), 2))

    # Run metrics on Random Forest Classifier
    rf = RandomForestClassifier(random_state = 777)
    acc, prec, re = xval_classifier(rf, reduced_features, labels, sss)
    rf_accuracy.append(round(float(acc), 2))
    rf_precision.append(round(float(prec), 2))
    rf_recall.append(round(float(re), 2))
    print('Feature:', features_list[i])
    print('Time Spent Fitting k = {0}: {1}s'.format(i + 1, round(time() - t, 2)))
    print('AdaBoost Metrics: \nAccuracy: {0}, Precision: {1}, Recall: {2}'.format(ada_accuracy[-1], 
          ada_precision[-1], ada_recall[-1]))
    print('RandomForest Metrics: \nAccuracy: {0}, Precision: {1}, Recall: {2}'.format(rf_accuracy[-1], 
          rf_precision[-1], rf_recall[-1]))
    scores.append({'k': str(i + 1), 'feature': features_list[i], 'ada_accuracy': ada_accuracy[-1], 
                  'ada_precision': ada_precision[-1],  'ada_recall': ada_recall[-1], 
                  'rf_accuracy': rf_accuracy[-1], 'rf_precision': rf_precision[-1], 
                  'rf_recall': rf_recall[-1]})
    if i < len(features[0]):
        print('Please continue to wait...')
    line()

# Average scores from each feature to get an idea of how the classifiers compare overall
line()
print('Average Scores For Each Classifier:')

avg_ada_accuracy = [] 
avg_ada_precision = [] 
avg_ada_recall = []
avg_rf_accuracy = []
avg_rf_precision = []
avg_rf_recall = []

for k in scores:
    avg_ada_accuracy.append(k['ada_accuracy'])
    avg_ada_precision.append(k['ada_precision'])
    avg_ada_recall.append(k['ada_recall'])
    avg_rf_accuracy.append(k['rf_accuracy'])
    avg_rf_precision.append(k['rf_precision'])
    avg_rf_recall.append(k['rf_recall'])

print('Average AdaBoost Accuracy:', round(sum(avg_ada_accuracy) / len(avg_ada_accuracy), 2))
print('Average AdaBoost Precision:', round(sum(avg_ada_precision) / len(avg_ada_precision), 2))
print('Average AdaBoost Recall:', round(sum(avg_ada_recall) / len(avg_ada_recall), 2))
print('Average RandomForest Accuracy:', round(sum(avg_rf_accuracy) / len(avg_rf_accuracy), 2))
print('Average RandomForest Precision:', round(sum(avg_rf_precision) / len(avg_rf_precision), 2))
print('Average RandomForest Recall:', round(sum(avg_rf_recall) / len(avg_rf_recall), 2))
#line()
#print('Show score values for each feature')
#pprint.pprint(scores)

#Create plots to visualize what the scores of each feature look like
line()
print('Plot metrics for each classifier, showing the values of each k feature')
line()
ada_df = pd.DataFrame({'ada_accuracy': ada_accuracy, 'ada_precision': ada_precision, 
         'ada_recall': ada_recall})
rf_df = pd.DataFrame({'rf_accuracy': rf_accuracy, 'rf_precision': rf_precision, 
        'rf_recall': rf_recall})
ada_df.plot()
rf_df.plot()
#plt.show()
#Hide plots

print('When comparing AdaBoost to RandomForest, AdaBoost has a  higher average recall,')
print('and a more stable (but lower) precison.')
print('RandomForest has a higher average precision, with more variation, and a slightly ')
print('higher accuracy.')
print('It appears that overall RandomForest is doing a bit of a better job than AdaBoost.')

line()
# Use SelectKBest to score and identify the best features in each classifier
print('SelectKBest Features:')

max_ada_recall = ada_recall.index(max(ada_recall)) + 1
max_ada_precision = ada_precision.index(max(ada_precision)) + 1
max_rf_recall = rf_recall.index(max(rf_recall)) + 1
max_rf_precision = rf_precision.index(max(rf_precision)) + 1

bfl_ada_recall, bf_ada_recall = best_features(max_ada_recall, features, labels, features_list)
bfl_rf_recall, bf_rf_recall = best_features(max_rf_recall, features, labels, features_list)

print('Number of Ada Recall Best Features', len(bfl_ada_recall) - 1)
print('Number of RF Recall Best Features:', len(bfl_rf_recall) - 1)
print('AdaBoost has many more features that end up on the best features list than RandomForest')

line()
print('SelectKBest Features - Ada: {0}'.format(len(bfl_ada_recall) - 1))
for f in bfl_ada_recall[1:]:
    print(f + ' with a score of: {0}'.format(round(sel.scores_[bfl_ada_recall[1:].index(f)], 2)))
line()

RF = RandomForestClassifier(random_state = 777)
RF.fit(bf_rf_recall, labels)
print('RF Feature Importance:')
print(RF.feature_importances_)

print('Best Features - RF:')
for f in bfl_rf_recall[1:]:
    print(f + ' with a score of: {0}'.format(round(sel.scores_[bfl_rf_recall[1:].index(f)], 2)))
line()
print('With AdaBoost and RandomForest, there are 3 best features we can examine/tune.')
print('Precision for both classifiers is already over .3, however, RandomForest has a much')
print('lower average recall')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
#rf_tuning_parameters = {'n_estimators': [90], 'min_samples_split': [3], 'max_features': [1]}
rf_tuning_parameters = {'n_estimators': [125], 
                        'min_samples_split': [2], 'max_features': [3]}
#ada_tuning_parametrers = {'n_estimators': [70], 'learning_rate': [.6]}
ada_tuning_parameters = {'n_estimators': [100],
                        'learning_rate': [1]}

ada_clf = tune_AdaBoost(ada_tuning_parameters, sss, bf_ada_recall, labels)
test_classifier(ada_clf, my_dataset, bfl_ada_recall, folds = 1000)
#print(xval_classifier(ada_clf, bf_ada_recall, labels, sss))


# Tune RandomForest
rf_clf = tune_RandomForest(rf_tuning_parameters, sss, bf_rf_recall, labels)
test_classifier(rf_clf, my_dataset, bfl_rf_recall, folds = 1000)
#print(xval_classifier(rf_clf, bf_rf_recall, labels, sss))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#Dump Randomforest 
dump_classifier_and_data(rf_clf, my_dataset, bfl_rf_recall)