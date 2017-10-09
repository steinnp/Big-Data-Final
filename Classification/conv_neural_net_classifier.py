import sys, os
foo_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../DataGathering', '..')))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../Classification', '..')))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../TextCleaning', '..')))
sys.path.insert(0, '../DataGathering/')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import DataGathering.trainingdata as td
import numpy as np
from keras import optimizers as opt
from keras.layers import Dense, Embedding, Input, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.models import Model, Sequential
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, f1_score, accuracy_score

MAX_SEQ_LENGTH = 400
EMBEDDING_DIM = 20


def preprocess_text_for_cnn_keras(train, val):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train)  # fit only on train.

    train_sequences = tokenizer.texts_to_sequences(train)
    val_sequences = tokenizer.texts_to_sequences(val)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    train = pad_sequences(train_sequences, MAX_SEQ_LENGTH)
    val = pad_sequences(val_sequences, MAX_SEQ_LENGTH)

    return train, val, word_index


def get_uniform_distribution(x_data, y_data, chunk_size=1000):
    counts = np.zeros(5)
    new_x = []
    new_y = []
    for i, x in enumerate(x_data):
        if counts[int(y_data[i] - 1)] < chunk_size:
            counts[int(y_data[i] - 1)] += 1
            new_x.append(x)
            new_y.append(y_data[i])
    return new_x, new_y



def get_amazon_data():
    x_data = open(r'C:\Users\steinnp\Desktop\KTH\Big-Data\dataset_for_class\dataset_for_class\2m_x_train_set.txt', 'r').readlines()
    y_data = [float(x.strip()) for x in open(r'C:\Users\steinnp\Desktop\KTH\Big-Data\dataset_for_class\dataset_for_class\2m_y_train_set.txt', 'r').readlines()]
    x_data, y_data = get_uniform_distribution(x_data, y_data, 5000)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_data)
    sequences = tokenizer.texts_to_sequences(x_data)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, MAX_SEQ_LENGTH)

    lol = []
    for l in y_data:
        if int(l) == 3:
            lol.append(1)
        elif int(l) > 3:
            lol.append(2)
        else:
            lol.append(0)
    labels = to_categorical(np.asarray(lol))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(0.8 * data.shape[0])

    x_train = data[:nb_validation_samples]
    y_train = labels[:nb_validation_samples]
    x_test = data[nb_validation_samples:]
    y_test = labels[nb_validation_samples:]
    return x_train, y_train, x_test, y_test, word_index

def build_ffnn_model(inp_shape, output_size):
    model = Sequential()
    model.add(Dense(256, input_shape=(inp_shape,), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(output_size, activation='softmax'))
    # model.add(Dense(1))

    rms = opt.RMSprop(lr=0.001)
    model.compile(optimizer=rms,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("model compiled")
    return model

def build_cnn_model(word_index, output_size):
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQ_LENGTH)

    sequence_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences = Dropout(0.2)(embedded_sequences)
    x = Conv1D(filters=10,
               strides=1,
               kernel_size=4,
               activation='relu',
               padding='valid')(embedded_sequences)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=10,
               strides=1,
               kernel_size=8,
               activation='relu',
               padding='valid')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    preds = Dense(output_size, activation='softmax')(x)

    model = Model(sequence_input, preds)
    rms = opt.RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rms,
                  metrics=['acc'])
    return model


def train_cnn():
    # training = td.TrainingData()
    # training.set_tweet_training_data()
    # training_reviews, training_ratings, validation_reviews, validation_ratings, train_raw, val_raw = training.get_training_validation_data(0.8)
    # y_train = training.tweets_to_amazon_ratings(training_ratings)
    # y_val = training.tweets_to_amazon_ratings(validation_ratings)
    # y_train = np.array(training_ratings)
    # y_val = np.array(validation_ratings)
    # training_reviews = training.matrix_to_dense(training_reviews)
    # training_reviews = training.matrix_to_dense(validation_reviews)
    # x_train, x_val, word_index = preprocess_text_for_cnn_keras(train_raw, val_raw)

    x_train, y_train, x_val, y_val, word_index = get_amazon_data()

    y_val_max = np.array([np.array(i).argmax(axis=0) for i in y_val])
    y_train_max = np.array([np.array(i).argmax(axis=0) for i in y_train])

    model = build_ffn_model(len(x_train[0]), len(y_train[0]))

    old_max = 0
    no_update_count = 0
    best_model = 0
    for i in range(200):
        print("NN fold: {}".format(i))
        model.fit(x_train, y_train,
                  epochs=1, batch_size=64)
        yp = model.predict(x_val)
        yp_max = np.array([i.argmax(axis=0) for i in yp])
        print(yp_max[0])
        print(y_val_max[0])
        f1 = f1_score(y_val_max, yp_max, average='weighted')
        if old_max < f1:
            best_model = model
            old_max = f1
            no_update_count = 0
        else:
            no_update_count += 1
            if no_update_count >= 20:
                break

        print("-----------------VALIDATION RESULTS------------------")
        report = classification_report(y_val_max, yp_max)
        conf = confusion_matrix(y_val_max, yp_max)
        accuracy = accuracy_score(y_val_max, yp_max)
        print(report)
        print(conf)
        print(accuracy)
    best_model.save("amazon_cnn_model.h5")
'''
        print("-----------------TRAIN RESULTS------------------")
        yp = model.predict(x_train)

        yp_max = np.array([i.argmax(axis=0) for i in yp])

        report = classification_report(y_train_max, yp_max)
        conf = confusion_matrix(y_train_max, yp_max)

        print(report)
        print(conf)
'''


def train_nn():
    training = td.TrainingData()
    training.set_tweet_training_data()
    trainingReviews, trainingRatings, validationReviews, validationRatings, train_raw, val_raw = training.get_training_validation_data(0.8)

    trainingReviews = training.matrix_to_dense(trainingReviews)
    validationReviews = training.matrix_to_dense(validationReviews)

    trainingReviews = np.reshape(trainingReviews, (len(trainingReviews), trainingReviews.shape[2]))
    validationReviews = np.reshape(validationReviews, (len(validationReviews), validationReviews.shape[2]))

    y_train = training.tweets_to_amazon_ratings(trainingRatings)
    y_val = training.tweets_to_amazon_ratings(validationRatings)

    y_val_max = np.array([np.array(i).argmax(axis=0) for i in y_val])

    y_train = np.array(y_train)

    model = build_ffnn_model(len(trainingReviews[0]), len(y_train[0]))
    for i in range(10):
        print("N fold {}".format(i))
        model.fit(trainingReviews, y_train, verbose=2,
                  epochs=10)

        yp = model.predict(validationReviews)
        yp_max = np.array([i.argmax(axis=0) for i in yp])
        print(classification_report(y_val_max, yp_max))

if __name__ == '__main__':
    train_nn()
