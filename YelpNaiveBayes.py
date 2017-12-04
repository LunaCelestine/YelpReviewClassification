# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:08:40 2017

@author: Bradley Dabdoub
"""
import gc
import time
import numpy
import pandas
from sklearn.cross_validation import train_test_split 
from collections import Counter
import re
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

start = time.clock()

df1 = pandas.read_csv('yelp_academic_dataset_review.csv', header=0, delimiter=',')#,encoding ='latin1'

#Select column 3, "text", store in reviews
reviews = df1.iloc[:, 3].values

#Select column 5, "stars", store in ratings
ratings = df1.iloc[:, 5].values
df1 = None
#If the star rating is greater than 3, make set the label as 1, otherwise set it to -1
ratings = numpy.where(ratings > 3, 1, -1)


#Train test split 70/15/15%
X_train, X_test_validation, Y_train, Y_test_validation = train_test_split(reviews, ratings, test_size=.3, random_state=1)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_test_validation, Y_test_validation, test_size=.5, random_state=1)

reviews, ratings = None


#These stopwords are a combination of the google stopwords and the stopwords provided in the notes with duplicates removed
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
            #reviewWords = review.split() 
            str(review)
            reviewWords = [w for w in review.split(" ") if w not in stopwords]
            for reviewWord in reviewWords:
                wordList.append(reviewWord)
    return wordList

posBag = bagOWords(X_train, Y_train, 1)
negBag = bagOWords(X_train, Y_train, -1)

negCounter = Counter(negBag)
posCounter = Counter(posBag)

#print(posCounter)
#print(negCounter == posCounter)

labelCount = Counter(Y_train)
negCount = labelCount.get(-1)
posCount = labelCount.get(1)

negClassProb = negCount / (negCount + posCount)
posClassProb = posCount / (negCount + posCount)
#print(negClassProb)
#print(posClassProb)



def NaiveBayes(review, count, prob):
    pred = 1
    reviewWords = Counter(re.split("\s+", review))
    for reviewWord in reviewWords:
        pred *= reviewWords.get(reviewWord) * ((count.get(reviewWord, 0) + 1) / sum(count.values()))
    pred *= prob
    return pred  
 
def predict(review):
    negativePrediction = NaiveBayes(review, negCounter, negClassProb)
    positivePrediction = NaiveBayes(review, posCounter, posClassProb)
    
    if negativePrediction > positivePrediction:
        return -1
    return 1

def main():
    predictions = []
    for review in X_validation:
        predictions.append(predict(review))    
    end = time.clock()
    duration = end - start

    print('Misclassified samples: %d' % (Y_validation != predictions).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_validation, predictions))
    print('F1 Score: ' +  str(f1_score(Y_validation, predictions)))
    print("---Duration: %s seconds ---" % duration)
    gc.collect()
    
          
main()





            
    





















