# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:08:40 2017

@author: Bradley Dabdoub
"""
import sys
import gc
import time
import numpy
import pandas
from sklearn.cross_validation import train_test_split 
from collections import Counter
import re
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
import math

numpy.set_printoptions(threshold=numpy.inf)


start = time.clock()
#IMPORTANT: Change nrows paramater to determine how many data samples you want in your whole set (or remove it for entire dataset)
df = pandas.read_csv('processed-reviews-ratings.csv', nrows=15000, header=0, delimiter=',', usecols=["text", "stars"])#,encoding ='latin1'
df = df.dropna(how='any', axis=0)

#Select column 3, "text", store in reviews
reviews = df.iloc[:, 0].values

#Select column 5, "stars", store in ratings
ratings = df.iloc[:, 1].values

#If the star rating is greater than 3, set the label to 1, otherwise set it to -1
ratings = numpy.where(ratings > 3, 1, -1)

#Train test split with random state 1, maintain proportion of labels in Y_sample, train/test_validation are 70/30% of original data
X_train, X_test_validation, Y_train, Y_test_validation, = train_test_split(reviews, ratings, test_size=.3, stratify=ratings, random_state=1)

#Train test split with random state 1, maintain proportion of labels in Y_test_validation, validation/test are 15/15% of original data
X_validation, X_test, Y_validation, Y_test = train_test_split(X_test_validation, Y_test_validation, test_size= .5, stratify=Y_test_validation, random_state=1)

stopwords = {'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at', 'be',
'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t',
'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has' 'hasn\'t',
'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how',
'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more',
'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves',
'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than',
'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll',
'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll',
'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s',
'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours',
'yourself', 'yourselves', ',', '.', '-', '!'}


def bagOWords(reviews, labels, target):
    wordList = []
    for index,review in enumerate(reviews):
        if (labels[index] == target):
            reviewWords = review.split() 
            for reviewWord in reviewWords:
                wordList.append(reviewWord)
    return wordList

posBag = bagOWords(X_train, Y_train, 1)
negBag = bagOWords(X_train, Y_train, -1)


negCounter = Counter(x for x in negBag if x not in stopwords)
posCounter = Counter(x for x in posBag if x not in stopwords)

# for words appearing more than once
# negCounter = {k:negCounter[k] for k in negCounter if negCounter[k] > 1}
# posCounter = {k:posCounter[k] for k in posCounter if posCounter[k] > 1}

labelCount = Counter(Y_train)
negCount = labelCount.get(-1)
posCount = labelCount.get(1)

negClassProb = negCount / (negCount + posCount)
posClassProb = posCount / (negCount + posCount)

def NaiveBayes(review, count, prob):
    pred = 1
    logProb = math.log(prob)
    reviewWords = Counter(re.split("\s+", review))

    for reviewWord in reviewWords:
        # print(pred *(reviewWords.get(reviewWord) * ((count.get(reviewWord, 0) + 1) / sum(count.values()))))
        pred += math.log(reviewWords.get(reviewWord) * ((count.get(reviewWord, 0) + 1) / sum(count.values())))
        # pred *= reviewWords.get(reviewWord) * ((count.get(reviewWord, 0) + 1) / sum(count.values()))
        if(pred == 0.0):
            sys.exit(0)
    
    pred += logProb
    return pred  

def predict(review):
    negativePrediction = NaiveBayes(review, negCounter, negClassProb)
    positivePrediction = NaiveBayes(review, posCounter, posClassProb)
    
    if negativePrediction > positivePrediction:
        return -1
    return 1

def main():
    predictions = []
    count = 0
    for review in X_validation:
        count = count +1
        
        predictions.append(predict(review))
    end = time.clock()
    duration = end - start
    print("Total Reviews:", count)

    misclass = numpy.where(Y_validation != predictions, 1, 0)
    print(metrics.classification_report(Y_validation, predictions, target_names=["1", "-1"]))

    print('Misclassified samples: %d' % misclass.sum())
    print('Accuracy: %.2f' % accuracy_score(Y_validation, predictions))
    print('F1 Score: ' +  str(f1_score(Y_validation, predictions, average=None)))
    print("---Duration: %s seconds ---" % duration)
    gc.collect()

main()





            
    





















