import sys, os
foo_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../DataGathering', '..')))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../Classification', '..')))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../TextCleaning', '..')))
sys.path.insert(0, '../DataGathering/')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras import optimizers as opt
from keras.layers import Dense, Embedding, Input, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.models import Model
import csv

MAX_SEQ_LENGTH = 400
EMBEDDING_DIM = 20


def load_model_from_File(word_index, output_size):
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
    model.load_weights(r'/Users/eysteinngunnlaugsson/Documents/KTHPeriod1/BigDataCourse/Big-Data-Final/cnn-weights-imporement-00-- 0.74.hdf5')
    rms = opt.RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rms,
                  metrics=['acc'])
    return model


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


def init_tokenizer_and_get_data(tweets):
    x_data = open(r'/Users/eysteinngunnlaugsson/Documents/KTHPeriod1/BigDataCourse/Big-Data-Final-local/Classification/amazon_data/2m_x_train_set.txt', 'r').readlines()
    y_data = [float(x.strip()) for x in open(r'/Users/eysteinngunnlaugsson/Documents/KTHPeriod1/BigDataCourse/Big-Data-Final-local/Classification/amazon_data/2m_y_train_set.txt', 'r').readlines()]
    x_data, y_data = get_uniform_distribution(x_data, y_data, 50000)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_data)
    sequences = tokenizer.texts_to_sequences(tweets)

    word_index = tokenizer.word_index
    data = pad_sequences(sequences, MAX_SEQ_LENGTH)

    return data, word_index


def predict_on_file(file_path):
    tweet_texts = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            tweet_texts.append(row[1])

    del tweet_texts[0]
    print('creating tokenizer')
    x_data, word_index = init_tokenizer_and_get_data(tweet_texts)
    print('done creating tokenizer')
    model = load_model_from_File(word_index, 3)
    yp = model.predict(x_data, verbose=1)
    yp_max = np.array([i.argmax(axis=0) for i in yp])
    return yp_max
