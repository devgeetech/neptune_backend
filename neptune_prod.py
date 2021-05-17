# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:43:24 2021

@author: joelg
"""

import requests
import os
import json
import time
from pandas.io.json import json_normalize
import spacy
import pandas as pd
import re
from collections import Counter
from datetime import datetime
nlp = spacy.load('en_core_web_md')
nlp2 = spacy.load('./content')

# INITIALISING VARIABLES
now = datetime.now()
event_list = {}

# GETTING TWITTER DATA
# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'


def auth():
    return os.environ.get("BEARER_TOKEN")


def create_url():
    query = "earthquake OR cyclone -is:retweet"
    tweet_fields = "tweet.fields=author_id,created_at,geo&max_results=10"
    url = "https://api.twitter.com/2/tweets/search/recent?query={}&{}".format(
        query, tweet_fields
    )
    return url


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def connect_to_endpoint(url, headers):
    response = requests.request("GET", url, headers=headers)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()
    
# PROCESSING TWITTER DATA
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAGYZOAEAAAAAD%2FG4jnISJ1wxC%2BdpB9gCjBHnxVM%3DZA8S9wRWY7AJdgQJ3lMCsaf05uhtTHxRtWDY6bIYx9hkB4nB4S'
url = create_url()
headers = create_headers(bearer_token)
i=0;
while i<10:
     # INITIALISING VARIABLES
     help_tweets = {}
     location_dictionary = {}
     # INITIALISING MAIN DATAFRAME
     json_response = connect_to_endpoint(url, headers)
     json_data_str = json.dumps(json_response, indent=4, sort_keys=True)
     json_data = json.loads(json_data_str)
     dset = json_normalize(json_data['data'])
     # CONSTRUCTING TWEET ONLY DATAFRAME
     newdset = dset.copy(deep=True)
     newdset.drop(['author_id','created_at','id','geo.coordinates.coordinates', 'geo.coordinates.type', 'geo.place_id' ], axis=1, inplace=True)
     
     # Cleaning Tweets
     def cleanTxt(text):
          text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions     
          text = re.sub('RT[\s]+', '', text) # Removing RT
          text = re.sub('#', '', text) # Removing RT
          text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
          return text
     newdset['text'] = newdset['text'].apply(cleanTxt)
     
     # Removing Stop Words
     all_stopwords = nlp.Defaults.stop_words
     tokenizer = nlp.tokenizer
     nlp.Defaults.stop_words |= {"\"",";","-","?",".","I","!",":","...","omfg","wtf","omg","sismo"} # updating stop word list
     for index, row in newdset.iterrows():
       tweet_as_tokens = tokenizer(newdset.loc[index].text.lower())
       tokens_without_sw= [token.text for token in tweet_as_tokens if not token.text in all_stopwords]
       newdset.loc[index].tweet = ' '.join(tokens_without_sw)
     
     # High-frequency words:
     most_important_topics = []
     TopicCount = Counter(" ".join(newdset.text).split()).most_common(15)
     for topic_name in TopicCount:
          most_important_topics.append(topic_name[0])
     print(most_important_topics)
     
     #LDA
     from sklearn.feature_extraction.text import CountVectorizer
     cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
     dtm=cv.fit_transform(newdset['text'])
     from sklearn.decomposition import LatentDirichletAllocation
     LDA = LatentDirichletAllocation(n_components=1, random_state=42)
     LDA.fit(dtm)
     for i, topic in enumerate(LDA.components_):
       print(f"The Top 15 words from given tweet list:")
       most_important_topics = most_important_topics + [cv.get_feature_names()[index] for index in topic.argsort()[-15:]]
       print(most_important_topics)
     
     # Assigning Event Type
     event_type = ""
     earthquake_found = any(elem in most_important_topics  for elem in ["earthquake", "earthquakes", "quake"]) 
     fire_found = any(elem in most_important_topics  for elem in ["fire", "wildfire", "flames"])
     cyclone_found = any(elem in most_important_topics  for elem in ["cyclone", "storm", "winds"])
     
     if earthquake_found:
       event_type = "earthquake"
     elif fire_found:
       event_type = "fire"
     elif cyclone_found:
          event_type = "cyclone"
     else:
          event_type="None"
     #if(event_type=="None"):
     #     return
     #print(event_type)
     
     # Semantic Analysis
     ground_truth_semantic = nlp("Earthquake")
     for index, row in newdset.iterrows():
       row_nlp = nlp(row.text)
       #print(ground_truth_semantic.similarity(row_nlp))
       if index==20:
         break
    
     #NER
     for index, row in newdset.iterrows():
       row_nlp = nlp(row.text)
       row_nlp2 = nlp2(row.text)
       if row_nlp2.ents:
           for ent in row_nlp2.ents:
                if ent.label_=="HLP":
                     help_tweets[index]={"tweet": "", "location": "", "timestamp": ""}
                     help_tweets[index]['tweet']=(dset.loc[index].text)
                     help_tweets[index]['timestamp']=(dset.loc[index].created_at)
       if row_nlp.ents:
         for ent in row_nlp.ents:
           if ent.label_=="GPE":
               loc_name = ent.text.lower()
               if index in help_tweets:
                    help_tweets[index]['location']=loc_name
               if loc_name not in location_dictionary:
                 location_dictionary[loc_name] = 0
               location_dictionary[loc_name] = location_dictionary[loc_name] + 1 
     
     #print(location_dictionary)
     most_frequent_locations = sorted(location_dictionary.items(), key = lambda kv:(kv[1], kv[0]))
     most_frequent_locations = most_frequent_locations[(len(most_frequent_locations)-3):]
     #print (most_frequent_locations)
     
     # BUILDING EVENT LIST
     for most_loc_i in most_frequent_locations:
          if event_type not in event_list.keys():
              event_list[event_type] = {most_loc_i[0]: most_loc_i[1]}      
          elif most_loc_i[0] in event_list[event_type]:
               event_list[event_type][most_loc_i[0]] = event_list[event_type][most_loc_i[0]] + most_loc_i[1]   
          else:
               event_list[event_type][most_loc_i[0]] = most_loc_i[1]
     #event_list[event_type] = most_frequent_locations
     
     # Most frequent loc from event_list
     event_loc_most = sorted(event_list[event_type].items(), key = lambda kv:(kv[1], kv[0]))
     event_loc_most = event_loc_most[(len(event_loc_most)-3):]
     
     # Splitting location names and frequencies
     loc_names = []
     loc_freqs = []
     for el in event_loc_most:
          loc_names.append(el[0])
          loc_freqs.append(str(el[1]))
     
     # Parsing help tweets to be compatible for GraphQL     
     help_tweet_str = ''  
     for h_twt in list(help_tweets.values()):
          help_tweet_str = "{orgStr}{{tweet:\\\"{rowTweet}\\\", location:\\\"{rowLoc}\\\", timestamp: \\\"{rowTime}\\\", status: \\\"1\\\" }},".format(orgStr=help_tweet_str, rowTweet=h_twt['tweet'], rowLoc=h_twt['location'], rowTime=datetime.strptime(h_twt['timestamp'], "%Y-%m-%d %H:%M:%S"))
     #print(help_tweet_str[:-1])

     #Saving data to MongoDB
     url = 'https://us-east-1.aws.realm.mongodb.com/api/client/v2.0/app/neptune_realm_1-uwjwy/graphql'
     headers = {
       'apiKey': 'tnpTbJUWMxGtVF5jHebxZRmjlI7Lxm45W0Gabc6diaVqs2IaVpzpm3Fg5gXzQu4A',
       'Content-Type': 'application/json'
     }
     #data = "\t{{\r\n\t  \t\"query\": \"mutation {{ updateOneCurrent_event( query: {{ event_type: \\\"{ev}\\\" }} set: {{  locations: [\\\"{loc2}\\\", \\\"{loc1}\\\", \\\"{loc0}\\\"], location_frequency: [\\\"{freq2}\\\", \\\"{freq1}\\\", \\\"{freq0}\\\"] }} ) {{ event_type locations location_frequency }} }}\",\r\n\t\t\"variables\": null\r\n\t}}".format(ev=event_type, loc2=loc_names[2], loc1=loc_names[1], loc0=loc_names[0], freq2=loc_freqs[2], freq1=loc_freqs[1], freq0=loc_freqs[0], hlp_twt_array=list(help_tweets.values()))
     data = "\t{{\r\n\t  \t\"query\": \"mutation {{ upsertOneCurrent_event( query: {{ event_type: \\\"{ev}\\\" }} data: {{  event_type: \\\"{ev}\\\", locations: [\\\"{loc2}\\\", \\\"{loc1}\\\", \\\"{loc0}\\\"], location_frequency: [\\\"{freq2}\\\", \\\"{freq1}\\\", \\\"{freq0}\\\"] }} ) {{ event_type locations location_frequency }} }}\",\r\n\t\t\"variables\": null\r\n\t}}".format(ev=event_type, loc2=loc_names[2], loc1=loc_names[1], loc0=loc_names[0], freq2=loc_freqs[2], freq1=loc_freqs[1], freq0=loc_freqs[0], hlp_twt_array=list(help_tweets.values()))
     data2 = "{{\"query\": \"mutation {{ insertManyHelp_tweets(data: [{hlp_twt_array}]){{ insertedIds}} }} \"}}".format(hlp_twt_array=help_tweet_str[:-1])
     
     response = requests.request(
       'POST',
       'https://us-east-1.aws.realm.mongodb.com/api/client/v2.0/app/neptune_realm_1-uwjwy/graphql',
       data=data,
       headers=headers,
     )
     
     response2 = requests.request(
      'POST',
       'https://us-east-1.aws.realm.mongodb.com/api/client/v2.0/app/neptune_realm_1-uwjwy/graphql',
       data=data2,
       headers=headers,
     )

     print("iteration {}".format(i))
     i = i + 1     
     
     time.sleep(300)


#def main():
#    bearer_token = 'AAAAAAAAAAAAAAAAAAAAAGYZOAEAAAAAD%2FG4jnISJ1wxC%2BdpB9gCjBHnxVM%3DZA8S9wRWY7AJdgQJ3lMCsaf05uhtTHxRtWDY6bIYx9hkB4nB4S'
#    url = create_url()
#    headers = create_headers(bearer_token)
#    json_response = connect_to_endpoint(url, headers)
#    json_data_str = json.dumps(json_response, indent=4, sort_keys=True)
#    json_data = json.loads(json_data_str)
#    df = json_normalize(json_data['data'])
#    print(df)
##    while True:
##         json_response = connect_to_endpoint(url, headers)
##         print(json.dumps(json_response, indent=4, sort_keys=True))
##         time.sleep(60)

#if __name__ == "__main__":
#    main()