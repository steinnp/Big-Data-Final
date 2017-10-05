import sys
sys.path.insert(0, '../DataGathering/')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import trainingdata as td
import numpy as np
from keras import optimizers as opt
from keras.layers import Dense, Embedding, Input, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

MAX_SEQ_LENGTH = 1600
EMBEDDING_DIM = 64


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
    x_data = open('amazon_data/2m_x_train_set.txt', 'r').readlines()
    y_data = [float(x.strip()) for x in open('amazon_data/2m_y_train_set.txt', 'r').readlines()]
    x_data, y_data = get_uniform_distribution(x_data, y_data)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_data)
    sequences = tokenizer.texts_to_sequences(x_data)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, MAX_SEQ_LENGTH)

    labels = to_categorical(np.asarray(y_data))
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


def build_cnn_model(word_index, output_size):
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQ_LENGTH)

    sequence_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences = Dropout(0.7)(embedded_sequences)
    x = Conv1D(filters=128,
               strides=1,
               kernel_size=3,
               padding='valid')(embedded_sequences)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=10,
               strides=1,
               kernel_size=4,
               padding='valid')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dropout(0.8)(x)
    x = Dense(50, activation='relu')(x)

    preds = Dense(output_size, activation='softmax')(x)

    model = Model(sequence_input, preds)
    rms = opt.RMSprop(decay=0.001)
    model.compile(loss='binary_crossentropy',
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

    model = build_cnn_model(word_index, len(y_train[0]))

    for i in range(200):
        print("NN fold: {}".format(i))
        model.fit(x_train, y_train,
                  epochs=1, batch_size=32)
        yp = model.predict(x_val)

        yp_max = np.array([i.argmax(axis=0) for i in yp])

        print("-----------------VALIDATION RESULTS------------------")
        report = classification_report(y_val_max, yp_max)
        conf = confusion_matrix(y_val_max, yp_max)
        print(report)
        print(conf)

        print("-----------------TRAIN RESULTS------------------")
        yp = model.predict(x_train)

        yp_max = np.array([i.argmax(axis=0) for i in yp])

        report = classification_report(y_train_max, yp_max)
        conf = confusion_matrix(y_train_max, yp_max)

        print(report)
        print(conf)

    model.save("amazon_cnn_model.h5")


if __name__ == '__main__':
    train_cnn()
