# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 09:58:44 2017

@author: ZeroTheHero
"""
import time
import numpy
import pandas
from sklearn.cross_validation import train_test_split 
from collections import Counter
import re
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# X = np.random.randint(5, size=(6, 100))
# y = np.array([1, 2, 3, 4, 5, 6])

# clf = MultinomialNB()
# clf.fit(X, y)
# MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
# print(clf.predict(X[2:3]))
# start = time.clock()

df = pandas.read_csv('processed-reviews-ratings.csv', nrows=3000000, header=0, delimiter=',', usecols=["text", "stars"])#,encoding ='latin1'

#Select column 3, "text", store in reviews
reviews = df.iloc[:, 0].values

#Select column 5, "stars", store in ratings
ratings = df.iloc[:, 1].values

#Train test split with random state 1, maintain proportion of labels in Y_sample, train/test_validation are 70/30% of original data
X_train, X_test_validation, Y_train, Y_test_validation, = train_test_split(reviews, ratings, test_size=.3, stratify=ratings, random_state=1)

#Train test split with random state 1, maintain proportion of labels in Y_test_validation, validation/test are 15/15% of original data
X_validation, X_test, Y_validation, Y_test = train_test_split(X_test_validation, Y_test_validation, test_size= .5, stratify=Y_test_validation, random_state=1)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, Y_train)

X_validate = count_vect.transform(X_validation)
X_validate_tfidf = tfidf_transformer.transform(X_validate)

predicted = clf.predict(X_validate_tfidf)

misclass = numpy.where(Y_validation != predicted, 1, 0)
# print("Misclassified: %d" + misclass.sum())
print('Misclassified samples: %d' % misclass.sum())
print('Accuracy: %.2f' % accuracy_score(Y_validation, predicted))
print('F1 Score: ' +  str(f1_score(Y_validation, predictions, average=None)))


