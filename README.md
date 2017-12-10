## Comp 488 Machine Learning Final Project
#### Yelp Dataset Review Classification Task
##### Team Members
- Bradley Dabdoub  
- Julia Cicale  
##### Project Report
Available at https://docs.google.com/document/d/1OTJbAaFZPt9gZ_ep6IyigVDv_GmxfLXOwXvKGA0Ca3U/edit?usp=sharing
##### Project Information
The scripts in this repository represent the Python code that we used to generate various ML models for classifying a Yelp review (1, 2, 3, 4, or 5 stars) based on the review's text content.  
The four scripts are used as follows:
- `DataPreProcessing.py` was used to convert the review text (`yelp_academic_dataset_review.csv`) into a usable format (all lowercase, no puncation except apostrophes, separated by spaces). It outputs a new csv called `processed-reviews-ratings.csv`.  
- `YelpNaiveBayes.py` is our own implementation of the Naive Bayes algorithm. Warning: running this with any sizable amount of data will take a long time.
- `scikitlearnYelpNaiveBayes.py` is our model using SciKitLearn's impelementation of Multinomial Naive Bayes (http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- `sklSVM.py` is our model using SciKitLearn's impelementation of SVM using SGD (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html). This is the model that ended up having the best performance, and was the one we used for our test dataset.

*Important: The scripts will not run without the dataset. It is available at https://www.yelp.com/dataset but must first be converted to a CSV to use with the first script, `DataPreProcessing.py`. If you would like to run the script and need a copy of the data in csv format, please contact either of us and we will be happy to upload to DropBox (the file is too large to upload to this repository)* 
