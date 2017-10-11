import sys
sys.path.insert(0, '../DataGathering/')
import DataGathering.trainingdata as td
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import NuSVC, LinearSVC, LinearSVR, SVC
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.pipeline import Pipeline
stemmer = LancasterStemmer()


def cont_to_disc(y_set):
    new_y = []
    for y in y_set:
        if y > 0.5:
            new_y.append(2)
        elif y < -0.5:
            new_y.append(0)
        else:
            new_y.append(1)
    return new_y

def train_and_predict(tweets):
    training = td.TrainingData()
    training.set_tweet_training_data()
    trainingReviews, trainingRatings, validationReviews, validationRatings, train_raw, val_raw = training.get_training_validation_data(1)

    y_train = cont_to_disc(trainingRatings)
    y_val = cont_to_disc(validationRatings)

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         # ('clf', NuSVC(kernel='sigmoid', nu=0.15, coef0=0.7)),
                         ('clf', LinearSVC()),
                         # ('clf', NuSVC(kernel='rbf', nu=0.15, coef0=0.7)),
                         # ('clf', SVC()),
                         ])

    text_clf.fit(train_raw, y_train)
    predicted = text_clf.predict(tweets)
    return predicted
    # text_clf.fit(parsedReviews, ratings)
    # print(classification_report(y_val, predicted))
    # print(confusion_matrix(y_val, predicted))
    # print(f1_score(y_val, predicted, average='weighted'))
