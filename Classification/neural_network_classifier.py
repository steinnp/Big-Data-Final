import sys, os
foo_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../DataGathering', '..')))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../Classification', '..')))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../TextCleaning', '..')))
sys.path.insert(0, '../DataGathering/')
import DataGathering.trainingdata as td
import os, sys
import numpy as np
from Classification.classification_utilities import save_keras_model, load_keras_model, classify_trump_tweets, classify_trump_tweets_one_at_a_time
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, f1_score
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

def build_ffnn_model(inp_shape, output_shape):
    model = Sequential()
    model.add(Dense(256, input_shape=(inp_shape,), activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(output_shape, activation="softmax"))
    #model.add(Dense(1))
   # model.load_weights(r"C:\Users\steinnp\Desktop\KTH\Big-Data\Big-Data-Final\ffnn-final-model.hdf5")
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("model compiled")
    return model

def cont_to_disc(y_set):
    new_y = []
    for y in y_set:
        if y == 3:
            new_y.append(1)
        elif y > 3:
            new_y.append(2)
        else:
            new_y.append(0)
    return new_y

def train_nn():
    training = td.TrainingData()
    #training.set_tweet_training_data()
    training.set_amazon_training_data(category_size=50000)
    trainingReviews, trainingRatings, validationReviews, validationRatings, train_raw, val_raw = training.get_training_validation_data(0.8)
    y_train = cont_to_disc(trainingRatings)
    y_val = cont_to_disc(validationRatings)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    trainingReviews = training.matrix_to_dense(trainingReviews)
    validationReviews = training.matrix_to_dense(validationReviews)

    trainingReviews = np.reshape(trainingReviews, (len(trainingReviews), trainingReviews.shape[2]))
    validationReviews = np.reshape(validationReviews, (len(validationReviews), validationReviews.shape[2]))
    model = build_ffnn_model(len(trainingReviews[0]), len(y_train[0]))
    #model = load_keras_model('FFNN-Regression.h5')
    val_max = [np.array(i).argmax(axis=0) for i in y_val]
    filepath = "weights-imporement-{epoch:02d}--{val_acc: .2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    callback_list = [checkpoint]
    for i in range(10):
        print("N fold {}".format(i))
        model.fit(trainingReviews, np.array(y_train),
                 epochs=2, validation_data=(validationReviews, np.array(y_val)),
                 callbacks=callback_list)
        yp = model.predict(validationReviews, verbose=1)
        yp_max = np.array([i.argmax(axis = 0) for i in yp])
        print(classification_report(val_max, yp_max))
        print(confusion_matrix(val_max, yp_max))
        print(f1_score(val_max, yp_max, average='weighted'))

    save_keras_model(model, 'FFNN-Regression.h5')


if __name__ == '__main__':
    # get an absolute path to the directory that contains mypackage
    foo_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
    sys.path.append(os.path.normpath(os.path.join(foo_dir, '../DataGathering', '..')))
    sys.path.append(os.path.normpath(os.path.join(foo_dir, '../Classification', '..')))
    sys.path.append(os.path.normpath(os.path.join(foo_dir, '../TextCleaning', '..')))
    model = load_keras_model('FFNN-Regression.h5')
    classify_trump_tweets_one_at_a_time(model, '480k_trump.csv',
                                        results_to_file=True,
                                        results_file_name="480ktrump_results.csv")
    # train_nn()
