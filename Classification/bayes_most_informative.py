import nltk
import csv
import matplotlib.pyplot as plt

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def extract_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features


def plot_most_important_words(tweets, predicts):
    labels = []
    for la in labels:
        if la == 0:
            labels.append('neg')
        if la == 1:
            labels.append('neu')
        if la == 2:
            labels.append('pos')

    train = list(zip(tweets, labels))
    tweets = []

    for (words, sentiment) in train:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        tweets.append((words_filtered, sentiment))
        
    word_features = get_word_features(get_words_in_tweets(tweets))

    training_set = nltk.classify.apply_features(extract_features, tweets, word_features)

    clf = nltk.NaiveBayesClassifier.train(training_set)
    mostinf = clf.get_most_informative_features_with_values(20)
    mostinf = sorted(mostinf, key=lambda x: x[1])
    words = [i[0] for i in mostinf]
    values = [i[1] for i in mostinf]
    x_range = [i for i in range(len(words))]

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    colors = ['red' if v < 0 else 'green' for v in values]
    values = sorted([abs(n) for n in values])
    ax.barh(x_range, values, align='center', color=colors)
    ax.set_yticks(x_range)
    ax.set_yticklabels(words)
    ax.set_xlabel('Word impact')

    plt.title("Most informative features")
    plt.show()
