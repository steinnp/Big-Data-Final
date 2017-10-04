import sys
import random
import os
sys.path.insert(0, '../TextCleaning/')
from TextCleaning.preprocessor import Preprocessor
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
        inputData = self.__preprocessor.fit_transform(inputData)
        self.inputData = inputData
        self.outputData = outputData

    def get_training_data(self):
        return self.inputData, self.outputData

    def split_training_validation(self, inputData, outputData, split):
        zipped = list(zip(inputData, outputData))
        self.rand.shuffle(zipped)
        inputData, outputData = list(zip(*zipped))
        split = int(split * len(inputData))
        trainingInputData = inputData[:split]
        trainingOutputData = outputData[:split]
        validationInputData = inputData[split:]
        validationOutputData = outputData[split:]
        return trainingInputData, trainingOutputData, validationInputData, validationOutputData

    def get_training_validation_data(self, split):
        return self.split_training_validation(self.inputData, self.outputData, split)


    def get_file_lines(self, filepath):
        lines = []
        fn = os.path.join(os.path.dirname(__file__), filepath)
        with open(fn, 'r', encoding='utf-8') as train_file:
            lines = train_file.read().splitlines()
        return lines

'''
training = TrainingData()
training.set_tweet_training_data()
trainingReviews, trainingRatings, validationReviews, validationRatings = training.get_training_validation_data(0.8)
print(len(trainingRatings))
print(trainingReviews[0].todense())
print(len(validationReviews))
'''
