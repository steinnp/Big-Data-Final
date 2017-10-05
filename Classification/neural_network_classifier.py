import sys
sys.path.insert(0, '../DataGathering/')
import trainingdata as td
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from keras.models import load_model
import csv


def save_keras_model(model, name):
    model.save(name)
    print("model saved")


def load_keras_model(name):
    model = load_model(name)
    print('model loaded')
    return model


def classify_trump_tweets(model, results_to_file=True, results_file_name="trump_results.csv"):
    """
    Classifies tweets about Donald Trump
    :param model: Keras model
    :param results_to_file: Boolean. Should the results be written to file
    :param results_file_name: Filename for if they should be written
    :return: The predictions for each trump tweet
    """
    training = td.TrainingData()
    training.set_tweet_training_data()

    tweet_texts = []
    with open('tset1_20k_Trump.csv', 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            tweet_texts.append(row[1])
    del tweet_texts[0]
    tweet_texts = [t.replace('\n', '') for t in tweet_texts]
    test_data = training._TrainingData__preprocessor.transform(tweet_texts)
    test_data = training.matrix_to_dense(test_data)

    test_data = np.reshape(test_data, (len(test_data), test_data.shape[2]))

    yp = model.predict(test_data)
    if results_to_file:
        with open(results_file_name, 'w', encoding='utf-8') as f:
            w = csv.writer(f, delimiter=',')
            for row in zip(tweet_texts, yp):
                w.writerow([row[0], row[1][0]])
    return zip(tweet_texts, yp)


def build_ffnn_model(inp_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=(inp_shape,), activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dropout(0.5))
    # model.add(Dense(len(trainingRatings[0])ax'))
    model.add(Dense(1))

    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mse'])
    print("model compiled")
    return model


def train_nn():
    training = td.TrainingData()
    training.set_tweet_training_data()
    trainingReviews, trainingRatings, validationReviews, validationRatings, train_raw, val_raw = training.get_training_validation_data(0.8)

    trainingReviews = training.matrix_to_dense(trainingReviews)
    validationReviews = training.matrix_to_dense(validationReviews)

    trainingReviews = np.reshape(trainingReviews, (len(trainingReviews), trainingReviews.shape[2]))
    validationReviews = np.reshape(validationReviews, (len(validationReviews), validationReviews.shape[2]))

    #model = build_ffnn_model(len(trainingReviews[0]))
    model = load_keras_model('FFNN-Regression.h5')
    val_max = [np.array(i).argmax(axis=0) for i in validationRatings]

    for i in range(10):
        print("N fold {}".format(i))
        model.fit(trainingReviews, np.array(trainingRatings), verbose=2,
                  epochs=10)
        yp = model.predict(validationReviews)
        print("test loss: {}".format(mean_squared_error(yp, validationRatings)))

    save_keras_model(model, 'FFNN-Regression.h5')


if __name__ == '__main__':
    #classify_trump_tweets()
    train_nn()









