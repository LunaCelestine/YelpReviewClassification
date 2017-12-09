# -*- coding: utf-8 -*-
"""
Spyder Editor

Used this script to scrub the data. The output is a csv called 'processed-review-ratings.csv' with 
two columns: the review, as a string with all words separated by spaces and all punctuation removed 
except apostrophes (e.g. "I'm" or "they're") and the review's star rating from 1-5
"""
import numpy
import pandas
import csv
import re
numpy.set_printoptions(threshold=numpy.inf)

df = pandas.read_csv('yelp_academic_dataset_review.csv', header=0, delimiter=',', usecols=["text", "stars"])#,encoding ='latin1'
#print(df)
df = df.dropna(how='any', axis=0)

for i, row in df.iterrows():
	processedString = row['text']
	processedString = re.sub(r'[^a-zA-Z\']+', ' ', processedString)
	df.set_value(i,'text',processedString)

df.to_csv('processed-reviews-ratings.csv', index=False)
