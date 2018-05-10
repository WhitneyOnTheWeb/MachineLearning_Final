#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "C:/JupyterNotebook/Machine_Learning/ud120-projects/" )
from sklearn.feature_extraction import stop_words
from tools.parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("text_learning/from_sara.txt", "r")
from_chris = open("text_learning/from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        #temp_counter += 1
        if temp_counter < 200:
            path = os.path.join('C:/JupyterNotebook/Machine_Learning/', path[:-1])
                #print(path)
            email = open(path, "r")

                ### use parseOutText to extract the text from the opened email
            email_temp = parseOutText(email)

                ### use str.replace() to remove any instances of the words
            signature = re.compile(r'(sara|shackleton|chris|germani|sshacklensf|cgermannsf)', re.I)
            extracted_text = signature.sub("", email_temp)
            word_data.append(extracted_text)
            #rem = ["sara", "shackleton", "chris", "germani"]
            #for word in rem:
            #    email_temp = email_temp.replace(word, '')
            #word_data.append(email_temp)

                ### append the text to word_data
            if name == 'sara':
                from_data.append(0)
                ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            else:
                from_data.append(1)

            email.close()

print(word_data[152])
print("emails processed")
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "wb") )
pickle.dump( from_data, open("your_email_authors.pkl", "wb") )


### in Part 4, do TfIdf vectorization here
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(stop_words = 'english', lowercase = True)
vec.fit_transform(word_data)
vocab_list = vec.get_feature_names()
print(len(vocab_list))
print(vocab_list[34597])
