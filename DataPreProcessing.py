# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy
import pandas
import csv
import re
df = pandas.read_csv('yelp_academic_dataset_review2.csv', header=0, delimiter=',', usecols=["text", "stars"])#,encoding ='latin1'
#print(df)
df = df.dropna(how='any', axis=0)
print(df)
reviews = df.iloc[:, 0].values
ratings = df.iloc[:, 1].values
print(ratings)
print(reviews)
for review in reviews:    
     review = re.sub(r'[^a-zA-Z\']+', '', review)

string = "I'm a review"
with open('yelp3.csv', 'w', newline='') as csvfile:
    csvWriter = csv.writer(csvfile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for review in reviews:
        strippedReview = review = re.sub(r'[^a-zA-Z\']+', '', review)
        csvWriter.writerow([strippedReview,#somehow the rating needs to get here ])
        