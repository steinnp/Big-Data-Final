import sys
import random
import os
import numpy as np
sys.path.insert(0, '../TextCleaning/')
# from TextCleaning.preprocessor import Preprocessor
from preprocessor import Preprocessor
from pprint import pprint

class TrainingData:
    rand = random.Random()

    def __init__(self, randomSeed = None):
        self.__preprocessor = Preprocessor()
        if randomSeed != None:
            self.rand = random.Random(randomSeed)

    def set_tweet_training_data(self):
        lines = self.get_file_lines('tweets_GroundTruth.txt')
        lines = [l.split('\t') for l in lines]
        outputData, inputData = zip(*[(float(l[1]), l[2]) for l in lines])
        self.rawInputData = inputData
        inputData = self.__preprocessor.fit_transform(inputData)
        self.inputData = inputData
        self.outputData = outputData

    def get_training_data(self):
        return self.inputData, self.outputData, self.rawInputData

    def split_training_validation(self, rawInputData, inputData, outputData, split):
        zipped = list(zip(rawInputData, inputData, outputData))
        self.rand.shuffle(zipped)
        rawInputData, inputData, outputData = list(zip(*zipped))
        split = int(split * len(inputData))
        trainingRawInputData = rawInputData[:split]
        trainingInputData = inputData[:split]
        trainingOutputData = outputData[:split]
        validationRawInputData = rawInputData[split:]
        validationInputData = inputData[split:]
        validationOutputData = outputData[split:]
        return trainingInputData, trainingOutputData, validationInputData, validationOutputData, trainingRawInputData, validationRawInputData

    def get_training_validation_data(self, split):
        return self.split_training_validation(self.rawInputData, self.inputData, self.outputData, split)

    @staticmethod
    def discretize_ratings(ratings):
        """
        Changes the ratings from continous values to an One-Hot array
        [1, 0, 0] = Negative
        [0, 1, 0] = Neutral
        [0, 0, 1] = Positive
        :return:
        """
        all_ratings = []
        for rating in ratings:
            if rating < -0.5:
                all_ratings.append([1, 0, 0])
            elif rating > 0.5:
                all_ratings.append([0, 0, 1])
            else:
                all_ratings.append([0, 1, 0])
        return all_ratings

    @staticmethod
    def tweets_to_amazon_ratings(ratings):
        """
        Changes the ratings from continous values to an One-Hot array
        [1, 0, 0, 0, 0] = Very Negative
        [0, 1, 0, 0, 0] = Moderately Negative
        [0, 0, 1, 0, 0] = Neutral
        [0, 0, 0, 1, 0] = Moderately Positive
        [0, 0, 0, 0, 1] = Very Positive
        :return:
        """
        all_ratings = []
        for rating in ratings:
            if rating > 2.4:
                all_ratings.append([0, 0, 0, 0, 1])
            elif rating > 0.8:
                all_ratings.append([0, 0, 0, 1, 0])
            elif rating > -0.8:
                all_ratings.append([0, 0, 1, 0, 0])
            elif rating > -2.4:
                all_ratings.append([0, 1, 0, 0, 0])
            else:
                all_ratings.append([1, 0, 0, 0, 0])
        return all_ratings

    def get_file_lines(self, filepath):
        fn = os.path.join(os.path.dirname(__file__), filepath)
        with open(fn, 'r', encoding='utf-8') as train_file:
            lines = train_file.read().splitlines()
        return lines

    def matrix_to_dense(self, reviews):
        """
        Changes a sparse matrix of multiple reviews
        to numpy arrays for each review
        :param reviews:
        :return:
        """
        return np.array([np.array(tr.todense()) for tr in reviews])

'''
training = TrainingData()
training.set_tweet_training_data()
trainingReviews, trainingRatings, validationReviews, validationRatings = training.get_training_validation_data(0.8)
print(len(trainingRatings))
print(trainingReviews[0].todense())
print(len(validationReviews))
'''
