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
    query = "-is:retweet lang:en (earthquake OR cyclone)"
    tweet_fields = "tweet.fields=author_id,created_at&max_results=100"
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
bearer_token = 'ENTER_BEARER_TOKEN_HERE'
url = create_url()
headers = create_headers(bearer_token)
loop_i=0;
while loop_i<10:
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
     newdset.drop(['author_id','created_at','id'], axis=1, inplace=True)
     
     # Cleaning Tweets
     def cleanTxt(text):
          text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions     
          text = re.sub('RT[\s]+', '', text) # Removing RT
          text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
          text = re.sub('#', '', text) # Removing RT
          text = re.sub('-', '', text) # Removing symbol
          text = re.sub(',', '', text) # Removing symbol
          text = re.sub('\.', '', text) # Removing symbol
          text = re.sub('\(', '', text) # Removing symbol
          text = re.sub('\)', '', text) # Removing symbol
          text = re.sub(':', '', text) # Removing colon
          text = re.sub(';', '', text) # Removing semicolon
          text = re.sub('\"', '', text) # Removing backslash
          text = re.sub('/', '', text) # Removing slash
          text = re.sub('_', '', text) # Removing underscore
          return text
     newdset['text'] = newdset['text'].apply(cleanTxt)
     
     # Removing Stop Words
     all_stopwords = nlp.Defaults.stop_words
     tokenizer = nlp.tokenizer
     nlp.Defaults.stop_words |= {"?","I","!","...","omfg","wtf","omg","sismo","lol"} # updating stop word list
     for index, row in newdset.iterrows():
       tweet_as_tokens = tokenizer(newdset.loc[index].text)
       tokens_without_sw= [token.text for token in tweet_as_tokens if not token.text in all_stopwords]
       newdset.loc[index].text = ' '.join(tokens_without_sw)
     
     # High-frequency words:
     most_important_topics = []
     most_important_topics_with_freq = {}
     TopicCount = Counter(" ".join(newdset.text).lower().split()).most_common(15)
     for topic_name in TopicCount:
          most_important_topics.append(topic_name[0])
          most_important_topics_with_freq[topic_name[0]]=topic_name[1]
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
     earthquake_list = ["earthquake", "earthquakes", "quake"]
     earthquake_found = any(elem in most_important_topics  for elem in earthquake_list) 
     fire_list = ["fire", "wildfire", "flames"]
     fire_found = any(elem in most_important_topics  for elem in fire_list)
     cyclone_list = ["cyclone", "storm", "winds"]
     cyclone_found = any(elem in most_important_topics  for elem in cyclone_list)
     event_count = 0
     
     if earthquake_found:
          event_type = "earthquake"
          event_count = event_count + 1
     if fire_found:
          event_type = "fire"
          event_count = event_count + 1
     if cyclone_found:
          event_type = "cyclone"
          event_count = event_count + 1
     
     if event_count == 0:
          time.sleep(300)
          continue
     
     if event_count > 1:
          sorted_topic_list = sorted(most_important_topics_with_freq.items(), key = lambda kv:(kv[1], kv[0]))
          for sorted_topic in reversed(sorted_topic_list):
               if sorted_topic[0] in earthquake_list or sorted_topic[0] in fire_list or sorted_topic[0] in cyclone_list: 
                    event_type = sorted_topic[0]
                    break
          
     
     # Semantic Analysis
     ground_truth_semantic = nlp("Earthquake")
     for index, row in newdset.iterrows():
       row_nlp = nlp(row.text)
       #print(ground_truth_semantic.similarity(row_nlp))
       if index==20:
         break
    
     #NER
     for index, row in dset.iterrows():
       row_nlp = nlp(row.text)
       row_nlp2 = nlp2(row.text)
       if row_nlp2.ents:
           for ent in row_nlp2.ents:
                if ent.label_=="HLP":
                     help_tweets[index]={"tweet": "", "location": "", "timestamp": "", "tweet_id": ""}
                     help_tweets[index]['tweet']=(dset.loc[index].text)
                     help_tweets[index]['timestamp']=(dset.loc[index].created_at)
                     help_tweets[index]['tweet_id']=(dset.loc[index].id)
       if row_nlp.ents:
         for ent in row_nlp.ents:
           if ent.label_=="GPE":
               loc_name = ent.text.lower()
               if index in help_tweets:
                    help_tweets[index]['location']=loc_name
               if loc_name not in location_dictionary:
                 location_dictionary[loc_name] = 0
               location_dictionary[loc_name] = location_dictionary[loc_name] + 1 
     
     print(location_dictionary)
     most_frequent_locations = sorted(location_dictionary.items(), key = lambda kv:(kv[1], kv[0]))
     most_frequent_locations = most_frequent_locations[(len(most_frequent_locations)-5):]
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
     event_loc_most = event_loc_most[(len(event_loc_most)-5):]
     
     # Splitting location names and frequencies
     loc_names = []
     loc_freqs = []
     for el in event_loc_most:
          loc_names.append(el[0])
          loc_freqs.append(str(el[1]))     
     if len(loc_names) < 5:
          loc_name_i = len(loc_names)
          while loc_name_i < 5:
               loc_names.append("")
               loc_freqs.append(0) 
               loc_name_i = loc_name_i + 1
     
     # Parsing help tweets to be compatible for GraphQL     
     help_tweet_str = ''  
     for h_twt in list(help_tweets.values()):
          help_tweet_str = "{orgStr}{{tweet:\\\"{rowTweet}\\\", location:\\\"{rowLoc}\\\", timestamp: \\\"{rowTime}\\\", tweet_id: \\\"{tweet_id}\\\", status: \\\"1\\\" }},".format(orgStr=help_tweet_str, rowTweet=h_twt['tweet'], rowLoc=h_twt['location'], tweet_id=h_twt['tweet_id'], rowTime=datetime.strptime(h_twt['timestamp'], "%Y-%m-%dT%H:%M:%S.000Z"))
     print(help_tweet_str[:-1])

     #Saving data to MongoDB
     graphql_url = 'https://us-east-1.aws.realm.mongodb.com/api/client/v2.0/app/neptune_realm_1-uwjwy/graphql'
     graphql_headers = {
       'apiKey': 'tnpTbJUWMxGtVF5jHebxZRmjlI7Lxm45W0Gabc6diaVqs2IaVpzpm3Fg5gXzQu4A',
       'Content-Type': 'application/json',
       'charset':'utf-8'
     }
     #data = "\t{{\r\n\t  \t\"query\": \"mutation {{ updateOneCurrent_event( query: {{ event_type: \\\"{ev}\\\" }} set: {{  locations: [\\\"{loc2}\\\", \\\"{loc1}\\\", \\\"{loc0}\\\"], location_frequency: [\\\"{freq2}\\\", \\\"{freq1}\\\", \\\"{freq0}\\\"] }} ) {{ event_type locations location_frequency }} }}\",\r\n\t\t\"variables\": null\r\n\t}}".format(ev=event_type, loc2=loc_names[2], loc1=loc_names[1], loc0=loc_names[0], freq2=loc_freqs[2], freq1=loc_freqs[1], freq0=loc_freqs[0], hlp_twt_array=list(help_tweets.values()))
     data = "\t{{\r\n\t  \t\"query\": \"mutation {{ upsertOneCurrent_event( query: {{ event_type: \\\"{ev}\\\" }} data: {{  event_type: \\\"{ev}\\\", locations: [\\\"{loc4}\\\", \\\"{loc3}\\\", \\\"{loc2}\\\", \\\"{loc1}\\\", \\\"{loc0}\\\"], location_frequency: [\\\"{freq4}\\\", \\\"{freq3}\\\", \\\"{freq2}\\\", \\\"{freq1}\\\", \\\"{freq0}\\\"] }} ) {{ event_type locations location_frequency }} }}\",\r\n\t\t\"variables\": null\r\n\t}}".format(ev=event_type, loc4=loc_names[4], loc3=loc_names[3], loc2=loc_names[2], loc1=loc_names[1], loc0=loc_names[0], freq4=loc_freqs[4], freq3=loc_freqs[3], freq2=loc_freqs[2], freq1=loc_freqs[1], freq0=loc_freqs[0], hlp_twt_array=list(help_tweets.values()))
     data2 = "{{\"query\": \"mutation {{ insertManyHelp_tweets(data: [{hlp_twt_array}]){{ insertedIds}} }} \"}}".format(hlp_twt_array=help_tweet_str[:-1])
     
     response = requests.request(
       'POST',
       'https://us-east-1.aws.realm.mongodb.com/api/client/v2.0/app/neptune_realm_1-uwjwy/graphql',
       data=data.encode('utf-8'),
       headers=graphql_headers,
     )
     
     response2 = requests.request(
      'POST',
       'https://us-east-1.aws.realm.mongodb.com/api/client/v2.0/app/neptune_realm_1-uwjwy/graphql',
       data=data2.encode('utf-8'),
       headers=graphql_headers,
     )
     
     print(response2.json())
     

     print("iteration {}".format(loop_i))
     loop_i = loop_i + 1     
     
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