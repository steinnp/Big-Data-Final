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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score




from sklearn import linear_model                                                                                                                                              
from sklearn import datasets                                                                                                                                                  
from sklearn import metrics  

training = td.TrainingData()
training.set_tweet_training_data()
trainingReviews, trainingRatings, validationReviews, validationRatings, train_raw, val_raw = training.get_training_validation_data(0.8)
trainingRatings = td.tweets_to_amazon_ratings(trainingRatings)
validationRatings = td.tweets_to_amazon_ratings(validationRatings)
#importing the dataset
#dataset = pd.read_csv('/Users/yunuskocyigit/Desktop/KTH/Big Data in Media Technology/projectfinal/tset1_20k_Trump.csv', names=['liked','txt'])
#x = dataset.iloc[:, 1].values
#y = dataset.iloc[:, 1].values
df = pd.read_csv('/Users/yunuskocyigit/Desktop/KTH/Big Data in Media Technology/Big-Data-Final/DataGathering/tweets_GroundTruth.txt', sep='\t', names=['liked','txt'])
df.head()

#TF-IDF Vectorizer
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

#our dependent value
y=df.liked

#array of independent verable
x= vectorizer.fit_transform(df.txt)
#wordlist= vectorizer.get_stop_words(df.txt)
#number of observations
print (y.shape)
#number of unique words
print (x.shape)

#Arrange the train and test set if need to be splitted
# we have to define our sets in different files
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)


# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#Fitting classifier to the Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(trainingReviews, trainingRatings)

#train with naive bayes classifier
#clf = naive_bayes.MultinomialNB()
clf = naive_bayes.BernoulliNB()
clf.fit(x_train, y_train)


#testing our model
roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
#another test
cl.classify("Trump is an amazing!")

