# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
import sys
sys.path.insert(0, '../DataGathering/')
from keras.layers import Dense, Dropout
from keras.models import Sequential

from sklearn.metrics import confusion_matrix, classification_report
# from DataGathering import trainingdata as td
import trainingdata as td
import numpy as np


if __name__ == '__main__':
    training = td.TrainingData()
    training.set_tweet_training_data()
    trainingReviews, trainingRatings, validationReviews, validationRatings = training.get_training_validation_data(0.8)

    trainingReviews = training.matrix_to_dense(trainingReviews)
    validationReviews = training.matrix_to_dense(validationReviews)

    trainingReviews = np.reshape(trainingReviews, (len(trainingReviews), trainingReviews.shape[2]))
    validationReviews = np.reshape(validationReviews, (len(validationReviews), validationReviews.shape[2]))

    print(trainingReviews.shape)
    print(validationReviews.shape)

    trainingRatings = training.discretize_ratings(trainingRatings)
    validationRatings = training.discretize_ratings(validationRatings)
    val_max = [np.array(i).argmax(axis=0) for i in validationRatings]

    model = Sequential()
    model.add(Dense(64, input_shape=(len(trainingReviews[0]), )))
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Dropout(0.5))
    model.add(Dense(16))
    model.add(Dropout(0.4))
    model.add(Dense(len(trainingRatings[0]), activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    for i in range(10):
        print("N fold {}".format(i))
        model.fit(trainingReviews, trainingRatings, verbose=2,
                  epochs=10)

        yp = model.predict(validationReviews)
        yp_max = [i.argmax(axis=0) for i in yp]

        rep = classification_report(val_max, yp_max)
        print(rep)




