import nltk
import csv
import matplotlib.pyplot as plt

word_features = []

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def extract_features(document):
    document_words = set(document)
    features = {}
    global word_features
    for word in word_features:
        features[word] = (word in document_words)
    return features

def get_most_informative_features_with_values(clf, n=100):
    # Determine the most relevant features, and display them.
    cpdist = clf._feature_probdist
    to_return = []
    print('Most Informative Features')

    for (fname, fval) in clf.most_informative_features(n):
        def labelprob(l):
            return cpdist[l, fname].prob(fval)

        labels = sorted([l for l in clf._labels
                         if fval in cpdist[l, fname].samples()],
                        key=labelprob)
        if len(labels) == 1:
            continue
        l0 = labels[0]
        l1 = labels[-1]
        if cpdist[l0, fname].prob(fval) == 0:
            ratio = 'INF'
        else:
            ratio = float((cpdist[l1, fname].prob(fval) / cpdist[l0, fname].prob(fval)))
        if l0 == 'pos':
            ratio = ratio * -1
        to_return.append((fname, ratio))
    return to_return


def plot_most_important_words(tweets, predicts):
    labels = []
    new_tweets = []
    for i, la in enumerate(predicts):
        if la == 0:
            new_tweets.append(tweets[i])
            labels.append('neg')
        if la == 1:
            pass
        if la == 2:
            new_tweets.append(tweets[i])
            labels.append('pos')

    train = list(zip(new_tweets, labels))
    tweets = []
    for (words, sentiment) in train:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        tweets.append((words_filtered, sentiment))
    global word_features
    word_features = get_word_features(get_words_in_tweets(tweets))

    training_set = nltk.classify.apply_features(extract_features, tweets)
    # training_set = nltk.classify.apply_features(word_features, tweets)

    clf = nltk.NaiveBayesClassifier.train(training_set)
    mostinf = get_most_informative_features_with_values(clf, 20)
    # mostinf = clf.get_most_informative_features_with_values(20)
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
