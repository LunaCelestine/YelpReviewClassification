# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:14:17 2017

@author: owner
"""
import time
import numpy
import pandas
import re
start = time.clock()
df = pandas.read_csv('yelp_academic_dataset_review2.csv', header=0, delimiter=',', usecols=["text", "stars"])#,encoding ='latin1'
#print(df)
df = df.dropna(how='any', axis=0)
print(df)
reviews = df.iloc[:, 0].values
ratings = df.iloc[:, 1].values
print(ratings)
print(reviews)
for review in reviews:    
    print(re.sub(r'[^a-zA-Z\']+', '', review))


end = time.clock()
print("done")
print("---Duration: %s seconds ---" % str(end-start))




