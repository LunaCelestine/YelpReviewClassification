# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:14:17 2017

@author: owner
"""
import time
import numpy
import pandas

start = time.clock()
df1 = pandas.read_csv('yelp_academic_dataset_review.csv', nrows=100, delimiter=',')#,encoding ='latin1'
reviews = df1.iloc[:, 5].values
print(reviews)
end = time.clock()
print("done")
print("---Duration: %s seconds ---" % str(end-start))