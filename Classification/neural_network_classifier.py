import DataGathering.trainingdata as td
import os, sys
import numpy as np
from Classification.classification_utilities import save_keras_model, load_keras_model, classify_trump_tweets
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from keras.models import load_model


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
    # get an absolute path to the directory that contains mypackage
    foo_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
    sys.path.append(os.path.normpath(os.path.join(foo_dir, '../DataGathering', '..')))
    sys.path.append(os.path.normpath(os.path.join(foo_dir, '../Classification', '..')))
    sys.path.append(os.path.normpath(os.path.join(foo_dir, '../TextCleaning', '..')))
    train_nn()
