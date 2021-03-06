# -*- coding: utf-8 -*-
##
## DM2583 Big Data in Media Technology
## Final project
##
## Carlo Rapisarda
## carlora@kth.se
##
## Data Gathering
## Oct. 2, 2017
##

import requests
import json
import base64
import csv

##
## README
##
## These values *must* be kept secret:
## client_id:     X2gxqgESTyTPg8xzkssZn39pA
## client_secret: wSnehKghoRjtusDfGoA2KLg1J3VfMbtraXz1eriQPxwwLG1Iso
##
## Call get_access_token() to get the access token (and store it somewhere)
##


## Requests an access token for the given credentials
## Must *not* be called frequently! (or else Twitter might return a 403 randomly)
def get_access_token(client_id, client_secret):

    basic_auth_base64 = base64.b64encode(bytes("{}:{}".format(client_id, client_secret), "utf-8"))

    try:
        response = requests.post(
            url="https://api.twitter.com/oauth2/token",
            headers={
                "Authorization": "Basic {}".format(basic_auth_base64.decode()),
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            },
            data={ "grant_type": "client_credentials" },
        )
        if response.status_code != 200:
            print('Response HTTP Status Code: {}'.format(response.status_code), flush=True)
            return response.content
        else:
            return json.loads(response.content)["access_token"]
    except requests.exceptions.RequestException:
        print('HTTP Request failed', flush=True)
        return ""


## Requests tweets with the Search API, returns a parsed json with the entire dump
## Example object returned: {statuses: [ ...tweets ], search_metadata: { ...metadata }}
## Max 100 tweets per request, use get_search_results()
def get_search_results_paged(query, count, max_id, result_type, token):

    params={ "q": query, "count": count, "lang": "en", "tweet_mode": "extended" }

    if max_id is not None:
        params["max_id"] = max_id
    if result_type is not None:
        params["result_type"] = result_type
    
    try:
        response = requests.get(
            url="https://api.twitter.com/1.1/search/tweets.json",
            params=params,
            headers={ "Authorization": "Bearer {}".format(token) },
        )
        if response.status_code != 200:
            print('Response HTTP Status Code: {}'.format(response.status_code), flush=True)
            return {}
        else:
            return json.loads(response.content)
    except requests.exceptions.RequestException:
        print('HTTP Request failed', flush=True)
        return {}


## Calls get_search_results_paged as many times as needed to get the specified amount of tweets
## Example object returned: [ ...tweets ]
## Max 450 calls / 15m window (i.e. 45'000 tweets / 15m)
## Param result_type may be "mixed", "recent", or "popular"
def get_search_results(query, count, result_type, token, max_id = None):
    res = []
    tot = 0
    while tot < count:
        rem = count-tot
        print("Pulling {} tweets...".format(min(rem, 100)), flush=True)
        page = get_search_results_paged(query, min(rem, 100), max_id, result_type, token)
        if page == {}:
            print('An error has occurred! Returning partial results', flush=True)
            break
        tweets = page["statuses"]
        if len(tweets) == 0:
            print('No more tweets for the given query! Returning data collected up to now', flush=True)
            break
        last_tweet = tweets[len(tweets)-1]
        max_id = last_tweet["id"]-1
        tot += len(tweets)
        res += tweets
        print("{}/{} tweets pulled".format(tot, count), flush=True)
    return res


## Returns a copy filtering out some garbage from a given set of tweets
def reformat_tweets(tweets):
    result = []
    for t in tweets:
        result.append({
            "full_text": t["full_text"],
            "retweet_count": t["retweet_count"],
            "favorite_count": t["favorite_count"],
            "id": t["id_str"],
            "entities": t["entities"],
        })
    return result


## Converts (in-place) retweets in regular tweets
def convert_retweets(tweets):
    for i in range(0, len(tweets)):
        if "retweeted_status" in tweets[i]:
            tweets[i] = tweets[i]["retweeted_status"]


## Strips out (in-place) urls from full tweets
def strip_urls(full_tweets):
    for t in full_tweets:
        urls = []
        if "urls" in t["entities"]:
            for u in t["entities"]["urls"]:
                urls.append(u["url"])
        if "media" in t["entities"]:
            for m in t["entities"]["media"]:
                urls.append(m["url"])
        full_text = t["full_text"]
        for u in urls:
            full_text = full_text.replace(u, "")
        t["full_text"] = full_text


## Tweets to CSV
def tweets_to_csv(tweets, file_path):
    with open(file_path, 'w+', encoding='utf-8') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(["id", "full_text", "retweet_count", "favorite_count", "location", "lang", "date"])
        for t in tweets:
            writer.writerow([t["id"], t["full_text"], t["retweet_count"], t["favorite_count"], t["user"]["location"], t["lang"], t["created_at"]])
