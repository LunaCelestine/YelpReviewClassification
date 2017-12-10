import numpy
import pandas
from sklearn.cross_validation import train_test_split 
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

df = pandas.read_csv('processed-reviews-ratings.csv', nrows=500000, header=0, delimiter=',', usecols=["text", "stars"])#,encoding ='latin1'

#Select column 3, "text", store in reviews
reviews = df.iloc[:, 0].values

#Select column 5, "stars", store in ratings
ratings = df.iloc[:, 1].values

#Train test split with random state 1, maintain proportion of labels in Y_sample, train/test_validation are 70/30% of original data
X_train, X_test_validation, Y_train, Y_test_validation, = train_test_split(reviews, ratings, test_size=.3, stratify=ratings, random_state=1)

#Train test split with random state 1, maintain proportion of labels in Y_test_validation, validation/test are 15/15% of original data
X_validation, X_test, Y_validation, Y_test = train_test_split(X_test_validation, Y_test_validation, test_size= .5, stratify=Y_test_validation, random_state=1)


text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),
	('tfidf', TfidfTransformer(use_idf=True)),
	('clf', SGDClassifier(loss='hinge', penalty='l2',
		alpha=1e-3, random_state=42,
		max_iter=5, tol=None))])
text_clf = text_clf.fit(X_train, Y_train)
#trying out grid search!
#testing words, bigrams or 3grams, with or without idf, and with a penalty parameter of either 0.01 or 0.001 for the linear SVM

# parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1,3)], 'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3)}

# gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

# gs_clf = gs_clf.fit(X_train, Y_train)

# print(gs_clf.best_score_)
# for param_name in sorted(parameters.keys()):
# 	print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

##result of the above grid search: 
# clf__alpha: 0.001 
# tfidf__use_idf: True
# vect__ngram_range: (1, 3)

predicted = text_clf.predict(X_validation)
print(numpy.mean(predicted == Y_validation))
print(metrics.classification_report(Y_validation, predicted, target_names=["1", "2", "3", "4", "5"]))


misclass = numpy.where(Y_validation != predicted, 1, 0)
print('Misclassified samples: %d' % misclass.sum())
print('Accuracy: %.2f' % accuracy_score(Y_validation, predicted))
print('F1 Score: ' +  str(f1_score(Y_validation, predicted, average=None)))



