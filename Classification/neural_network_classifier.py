# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
import sys, os
print(os.getcwd())
sys.path.insert(0, '../DataGathering/')
import trainingdata as td
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error


def prep_data_for_nn_inout(trainingReviews, trainingRatings, validationReviews, validationRatings):
    trainingRatings = np.array(trainingRatings)
    validationRatings = np.array(validationRatings)

    trainingReviews = training.matrix_to_dense(trainingReviews)
    validationReviews = training.matrix_to_dense(validationReviews)

    trainingReviews = np.reshape(trainingReviews, (len(trainingReviews), trainingReviews.shape[2]))
    validationReviews = np.reshape(validationReviews, (len(validationReviews), validationReviews.shape[2]))
    return trainingReviews, trainingRatings, validationReviews, validationRatings

def build_ffnn_model(inp_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=(inp_shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Dropout(0.5))
    # model.add(Dense(len(trainingRatings[0])ax'))
    model.add(Dense(1))

    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mse'])
    return model


if __name__ == '__main__':
    training = td.TrainingData()
    training.set_tweet_training_data()
    trainingReviews, trainingRatings, validationReviews, validationRatings = prep_data_for_nn_inout(*training.get_training_validation_data(0.8))

    model = build_ffnn_model(len(trainingReviews[0]))

    val_max = [np.array(i).argmax(axis=0) for i in validationRatings]

    for i in range(20):
        print("N fold {}".format(i))
        model.fit(trainingReviews, np.array(trainingRatings), verbose=2,
                  epochs=10)

        yp = model.predict(validationReviews)
        train_yp = model.predict(trainingReviews)

        print("train loss: {}".format(mean_squared_error(train_yp, trainingRatings)))
        print("test loss: {}".format(mean_squared_error(yp, validationRatings)))






