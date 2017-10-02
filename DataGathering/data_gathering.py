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
            print('Response HTTP Status Code: {}'.format(response.status_code))
            return response.content
        else:
            return json.loads(response.content)["access_token"]
    except requests.exceptions.RequestException:
        print('HTTP Request failed')
        return ""


## Requests tweets with the Search API, returns a parsed json with the entire dump
## Example object returned: {statuses: [ ...tweets ], search_metadata: { ...metadata }}
## Max 100 tweets per request, can be paginated w/ proper adjustments
def get_search_results(query, count, token):

    try:
        response = requests.get(
            url="https://api.twitter.com/1.1/search/tweets.json",
            params={ "q": query, "count": count },
            headers={ "Authorization": "Bearer {}".format(token) },
        )
        if response.status_code != 200:
            print('Response HTTP Status Code: {}'.format(response.status_code))
            return {}
        else:
            return json.loads(response.content)
    except requests.exceptions.RequestException:
        print('HTTP Request failed')
        return {}


## Filters out some garbage from a given set of tweets
def reformat_tweets(tweets):
    result = []
    for t in tweets:
        result.append({
            "text": t["text"],
            "retweet_count": t["retweet_count"],
            "favorite_count": t["favorite_count"],
            "id": t["id_str"],
            "entities": t["entities"],
        })
    return result


# Some tests
# token = "AAAAAAAAAAAAAAAAAAAAAPY62gAAAAAAUGA8nJomXvOni%2FXpNNuvZhtgAMg%3DvtffumSGrVK3snSkAsWWIlyxNuL30DsaSM3yygdZOvZaYlqCc7"
# results = get_search_results("las vegas", 10, token)
# results = reformat_tweets(results["statuses"])
# print('Results: {}'.format(results))
# print(get_access_token("", ""))
