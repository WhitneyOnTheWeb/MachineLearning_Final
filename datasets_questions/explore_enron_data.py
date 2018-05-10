#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
from final_project.poi_email_addresses import poiEmails

pkl_file = "../ud120-projects/final_project/final_project_dataset_unix.pkl"
poi_file = open("../ud120-projects/final_project/poi_names.txt", 'r')
poi_names = poi_file.readlines()
poi_names = poi_names[2:]

enron_file_handler = open(pkl_file, "rb")
enron_data = pickle.load(enron_file_handler, fix_imports=True)
enron_file_handler.close()

enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
print(enron_data["SKILLING JEFFREY K"])
print(len(enron_data))
print(len(enron_data["SKILLING JEFFREY K"]))

emails = poiEmails()
poi_count = 0

#test poi count via email addresses
for f in enron_data:
    if enron_data[f]['poi']:
	    poi_count += 1
print(poi_count)

#count emails
print(len(emails))

#count poi names, minus header rows
print(len(poi_names))

   #stock belonging to James Prentice
#for key, value in enron_data.items():
#   print(key)
	
print(enron_data["PRENTICE JAMES"]['total_stock_value'])
print(enron_data["COLWELL WESLEY"]['from_this_person_to_poi'])
print(enron_data["SKILLING JEFFREY K"]['exercised_stock_options'])

execs = {'SKILLING JEFFREY K': '',
		 'FASTOW ANDREW S': '',
		 'LAY KENNETH L': ''}

for key in enron_data:
    if key in execs:
	    execs[key] = enron_data[key]['total_payments']

key, value = max(execs.items(), key = lambda x:x[1])		
print('Max Executive Salary:', key, ', ', value)

salary_count = 0 
for f in enron_data:
    if enron_data[f]['salary'] != 'NaN':
	    salary_count += 1
print(salary_count)

payments_count = 0 
for f in enron_data:
    if enron_data[f]['total_payments'] == 'NaN' and enron_data[f]['poi']:
	    payments_count += 1
print(payments_count)