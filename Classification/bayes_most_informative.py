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


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features


if __name__ == '__main__':
    tweet_texts = []
    with open('480k_trump.csv', 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in csv_reader:
            tweet_texts.append(row[1])
            count += 1
            if count == 200:
                break
    del tweet_texts[0]
    labels = ['pos' for _ in range(100)] + ['neg' for _ in range(100)]
    train = list(zip(tweet_texts, labels))
    tweets = []

    for (words, sentiment) in train:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        tweets.append((words_filtered, sentiment))
    word_features = get_word_features(get_words_in_tweets(tweets))

    training_set = nltk.classify.apply_features(extract_features, tweets)
    # training_set = nltk.classify.apply_features(word_features, tweets)

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
