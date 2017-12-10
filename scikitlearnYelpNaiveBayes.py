# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 09:58:44 2017

@author: ZeroTheHero
"""
import numpy
import pandas
from sklearn.cross_validation import train_test_split 
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

df = pandas.read_csv('processed-reviews-ratings.csv', nrows=3000000, header=0, delimiter=',', usecols=["text", "stars"])#,encoding ='latin1'

#Select column 3, "text", store in reviews
reviews = df.iloc[:, 0].values

#Select column 5, "stars", store in ratings
ratings = df.iloc[:, 1].values

#Train test split with random state 1, maintain proportion of labels in Y_sample, train/test_validation are 70/30% of original data
X_train, X_test_validation, Y_train, Y_test_validation, = train_test_split(reviews, ratings, test_size=.3, stratify=ratings, random_state=1)

#Train test split with random state 1, maintain proportion of labels in Y_test_validation, validation/test are 15/15% of original data
X_validation, X_test, Y_validation, Y_test = train_test_split(X_test_validation, Y_test_validation, test_size= .5, stratify=Y_test_validation, random_state=1)

text_clf = Pipeline([('vect', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	('clf', MultinomialNB())])
text_clf = text_clf.fit(X_train, Y_train)

predicted = text_clf.predict(X_validation)
print(numpy.mean(predicted == Y_validation))
print(metrics.classification_report(Y_validation, predicted, target_names=["1", "2", "3", "4", "5"]))


misclass = numpy.where(Y_validation != predicted, 1, 0)
print('Misclassified samples: %d' % misclass.sum())
print('Accuracy: %.2f' % accuracy_score(Y_validation, predicted))
print('F1 Score: ' +  str(f1_score(Y_validation, predicted, average=None)))


