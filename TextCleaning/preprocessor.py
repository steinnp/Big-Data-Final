import re
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import random
stemmer = LancasterStemmer()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from pprint import pprint

class Preprocessor:
    def __parse_review(self, reviewString):
        letters_only = re.sub('[^a-zA-Z]', ' ', reviewString) # replace non-letters
        letters_only = word_tokenize(letters_only)
        letters_only = " ".join(letters_only)
        lower_case = letters_only.lower()        # Convert to lower case
        words = lower_case.split()               # Split into words
        words = [stemmer.stem(w) for w in words if not w in stopwords.words("english")]
        return " ".join(words)

    def __init__(self):
        self.__vectorizer = TfidfVectorizer(stop_words = stopwords.words('english'), preprocessor = self.__parse_review)

    def get_vocabulary(self):
        return self.__vectorizer.vocabulary_

    def get_transformed_values(self):
        return self.__transformedValues

    def fit_transform(self, content):
        return self.__vectorizer.fit_transform(content)

    def transform(self, content):
        return self.__vectorizer.transform(content)
