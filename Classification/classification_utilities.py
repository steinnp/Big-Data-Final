from keras.models import load_model
import DataGathering.trainingdata as td
import numpy as np
import csv

def save_keras_model(model, name):
    model.save(name)
    print("model saved")


def load_keras_model(name):
    model = load_model(name)
    print('model loaded')
    return model


def classify_trump_tweets(model, path_to_tweets, results_to_file=True, results_file_name="trump_results.csv"):
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
    with open(path_to_tweets, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            tweet_texts.append(row[1])
    del tweet_texts[0]
    print("Tweets read from file")
    tweet_texts = [t.replace('\n', '') for t in tweet_texts]
    test_data = training._TrainingData__preprocessor.transform(tweet_texts)
    test_data = training.matrix_to_dense(test_data)

    test_data = np.reshape(test_data, (len(test_data), test_data.shape[2]))
    print("Tweet preprocessing done!")
    yp = model.predict(test_data)
    if results_to_file:
        with open(results_file_name, 'w', encoding='utf-8') as f:
            w = csv.writer(f, delimiter=',')
            for i, row in enumerate(zip(tweet_texts, yp)):
                if i % 10000 == 0:
                    print("Wrote {} tweets to file".format(i))
                w.writerow([row[0], row[1][0]])
    return zip(tweet_texts, yp)



def classify_trump_tweets_one_at_a_time(model, path_to_tweets, results_to_file=True, results_file_name="trump_results.csv"):
    """
    Classifies tweets about Donald Trump
    :param model: Keras model
    :param results_to_file: Boolean. Should the results be written to file
    :param results_file_name: Filename for if they should be written
    :return: The predictions for each trump tweet
    """
    training = td.TrainingData()
    training.set_tweet_training_data()
    count = 0
    tweet_ids = []
    with open(path_to_tweets, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        first_tweet = True
        for row in csv_reader:
            if first_tweet:
                first_tweet = False
                continue
            if count % 10000 == 0:
                print("Done {} tweets".format(count))
            count += 1
            t_id = row[0]
            t = row[1]
            t.replace('\n', '')
            test_data = training._TrainingData__preprocessor.transform([t])
            test_data = training.matrix_to_dense(test_data)
            test_data = np.reshape(test_data, (len(test_data), test_data.shape[2]))

            yp = model.predict(test_data)
            if results_to_file:
                with open(results_file_name, 'a', encoding='utf-8') as f:
                    w = csv.writer(f, delimiter=',')
                    w.writerow([t_id, yp[0][0]])

