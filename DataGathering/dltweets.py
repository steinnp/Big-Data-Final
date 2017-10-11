#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
##
## DM2583 Big Data in Media Technology
## Final project
##
## Carlo Rapisarda
## carlora@kth.se
##
## Tweets downloader
## Oct. 8, 2017
##

import argparse
from time import sleep
from data_gathering import get_search_results
from data_gathering import convert_retweets
from data_gathering import tweets_to_csv
from data_gathering import strip_urls

sleep_time = 5*60 # 5 minutes

DEF_OUTPUT_FOLDER = "./"
DEF_OUTPUT_FILENAME = "tweets"
DEF_N_TWEETS = 100
DEF_N_FILES = 1
DEF_TWEETS_TYPE = 'recent'
DEF_TOKEN = "AAAAAAAAAAAAAAAAAAAAAPY62gAAAAAAUGA8nJomXvOni%2FXpNNuvZhtgAMg%3DvtffumSGrVK3snSkAsWWIlyxNuL30DsaSM3yygdZOvZaYlqCc7"


parser = argparse.ArgumentParser(prog='Tweets downloader', usage='Some usage', description='Some description')
parser.add_argument('query', type=str, help='query used to search for tweets')
parser.add_argument('-d', '--dest',   help='destination folder', default=DEF_OUTPUT_FOLDER, type=str)
parser.add_argument('-n', '--name',   help='name of the file(s)', default=DEF_OUTPUT_FILENAME, type=str)
parser.add_argument('-k', '--token',  help='access token for Twitter', default=DEF_TOKEN, type=str)
parser.add_argument('-c', '--count',  help='number of tweets to download', default=DEF_N_TWEETS, type=int)
parser.add_argument('-s', '--fcount', help='number of files to save the results', default=DEF_N_FILES, type=int)
parser.add_argument('-t', '--type',   help='type of tweets to download', choices=['mixed', 'recent', 'popular'], default=DEF_TWEETS_TYPE)
parser.add_argument('--besteffort',   help='stops if there are no more tweets available', action="store_true")
args = parser.parse_args()

output_folder = args.dest
query = args.query
output_filename = args.name
n_tweets = args.count
n_files = args.fcount
tweets_type = args.type
token = args.token
besteffort = args.besteffort

tweets_per_file = int(n_tweets / n_files)
n_downloaded = 0
n_remaining = n_tweets
n_remaining_curr_file = tweets_per_file
results = []
max_id = None
besteffort_halt = False
i = 0

while i < n_files:
	
	results += get_search_results(query, n_remaining_curr_file, tweets_type, token, max_id)

	if len(results) != 0:
		last_tweet = results[len(results)-1]
		max_id = last_tweet["id"]-1
		n_downloaded += len(results)
		n_remaining -= len(results)
		n_remaining_curr_file -= len(results)

	if len(results) < tweets_per_file:
		if besteffort:
			print("Halting at best effort.", flush=True)
			besteffort_halt = True
		else:
			print("Retrying in {} seconds...".format(sleep_time), flush=True)
			sleep(sleep_time)
			continue

	convert_retweets(results)
	strip_urls(results)

	file_path = output_folder
	if not file_path.endswith("/"):
		file_path += "/"
	file_path += output_filename
	if n_files > 1:
		file_path += "_"+str(i)
	file_path += ".csv"
	
	tweets_to_csv(results, file_path)

	i += 1
	results = []
	n_remaining_curr_file = tweets_per_file

	if besteffort_halt:
		break

print("done", flush=True)
