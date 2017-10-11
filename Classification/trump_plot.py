
import sys, os
foo_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../DataGathering', '..')))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../Classification', '..')))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '../TextCleaning', '..')))
sys.path.insert(0, '../DataGathering/')
from svm_classifier import train_and_predict

import matplotlib.pyplot as plt
import datetime
import csv


def classify():
	tweet_texts = []
	tweets = []
	i = 0
	with open('../../480k_trump_merged.csv', 'r', encoding='utf-8') as csvfile:
	    csv_reader = csv.reader(csvfile, delimiter=',')
	    for row in csv_reader:
	        tweet_texts.append(row[1])
	        tweets.append(row)
	        i += 1
	        if i > 300000000:
	        	break
	    del tweet_texts[0]
	    del tweets[0]

	results = train_and_predict(tweet_texts)

	with open('../../480k_trump_classified.csv', 'w+', encoding='utf-8') as csvfile:
	    csv_writer = csv.writer(csvfile, delimiter=',')
	    for i in range(0, len(results)):
	    	csv_writer.writerow(tweets[i]+[results[i]])


def plot():
	values = []
	timestamps = []
	data = []

	with open('../../480k_trump_classified.csv', 'r', encoding='utf-8') as csvfile:
	    csv_reader = csv.reader(csvfile, delimiter=',')
	    for row in csv_reader:
	    	raw_date = row[6]
	    	value = row[7]
	    	timestamp = datetime.datetime.strptime(raw_date, "%a %b %d %H:%M:%S %z %Y").timestamp()
	    	values.append(value)
	    	timestamps.append(timestamp)
	    	data.append([timestamp, value])

	data = sorted(data, key=lambda x: x[0])

	maxTimestamp = data[-1][0]
	minTimestamp = data[0][0]

	bin_size = 1000*3600

	hours = int((maxTimestamp-minTimestamp)/bin_size)
	hourLists = [[] for _ in range(hours)]

	print(len(hourLists))

	for i in range(0,len(data)):
		h = int((data[i][0]-minTimestamp)/bin_size)-1
		print(h)
		hourLists[h].append(int(data[i][1]))


	values = []
		
	for hourList in hourLists:
		if len(hourList) >= 4:
			values.append(sum(hourList)/max(len(hourList), 1))

	plt.plot(range(0, len(values)), values, 'ro')
	plt.show()

#classify()
plot()
