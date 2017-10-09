# Naive Bayees for the project final
#importing the libraries
import os, sys
# get an absolute path to the directory that contains mypackage
foo_dir = os.path.dirname(os.path.join(os.getcwd(), '/Users/yunuskocyigit/Desktop/KTH/Big Data in Media Technology/Big-Data-Final/Classification/bayes_clasifier.py'))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../DataGathering', '..')))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../Classification', '..')))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../TextCleaning', '..')))
import pandas as pd
import DataGathering.trainingdata as td
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model                                                                                                                                              
from sklearn import datasets                                                                                                                                                  
from sklearn import metrics  

if __name__ == '__main__':
    training = td.TrainingData()
    training.set_tweet_training_data()
    trainingReviews, trainingRatings, validationReviews, validationRatings, train_raw, val_raw = training.get_training_validation_data(0.8)

    trainingRatings = [1 if t > 0 else 0 for t in trainingRatings]
    validationRatings = [1 if t > 0 else 0 for t in validationRatings]
    trainingReviews = np.array([t.todense() for t in trainingReviews])
    validationReviews = np.array([t.todense() for t in validationReviews])
    trainingReviews = np.reshape(trainingReviews, (trainingReviews.shape[0], trainingReviews.shape[2]))
    validationReviews = np.reshape(validationReviews, (validationReviews.shape[0], validationReviews.shape[2]))
    classifier = GaussianNB()
    classifier.fit(trainingReviews, trainingRatings)
    print("WHUUUAT")
    #train with naive bayes classifier
    #clf = naive_bayes.MultinomialNB()
    clf = naive_bayes.BernoulliNB()
    clf.fit(trainingReviews, trainingRatings)
    predicted = clf.predict(validationReviews)
    print(classification_report(validationRatings, predicted))


