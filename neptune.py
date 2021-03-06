import spacy
import pandas as pd
import re
import sys
from datetime import datetime
from collections import Counter
nlp = spacy.load('en_core_web_md')
nlp2 = spacy.load('./content')

#get time
now = datetime.now()

event_list = {}
help_tweets = {}
location_dictionary = {}
#dset = pd.read_csv('napa_earthquake_unaffected_filtered_hash_cleaned.txt', encoding='ISO-8859–1', delimiter='\t')
#dset = pd.read_csv('napa_earthquake_complete.txt', encoding='ISO-8859–1', delimiter='\t')
#dset = pd.read_csv('non_event.csv', encoding='ISO-8859–1')

#dset = pd.read_json('napa_earthquake_complete.json', encoding='ISO-8859–1')
dset = pd.read_json('napa_earthquake_unaffected_filtered_hash_cleaned.json', encoding='ISO-8859–1')

newdset = dset.copy(deep=True)
newdset.drop(['time','tweet_id','long','lat'], axis=1, inplace=True)

# Cleaning Tweets
def cleanTxt(text):
     text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions     
     text = re.sub('RT[\s]+', '', text) # Removing RT
     text = re.sub('#', '', text) # Removing RT
     text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
     return text
newdset['tweet'] = newdset['tweet'].apply(cleanTxt)

# Removing Stop Words
all_stopwords = nlp.Defaults.stop_words
tokenizer = nlp.tokenizer
nlp.Defaults.stop_words |= {"\"",";","-","?",".","I","!",":","...","omfg","wtf","omg","sismo"} # updating stop word list

for index, row in newdset.iterrows():
  tweet_as_tokens = tokenizer(newdset.loc[index].tweet.lower())
  tokens_without_sw= [token.text for token in tweet_as_tokens if not token.text in all_stopwords]
  newdset.loc[index].tweet = ' '.join(tokens_without_sw)
  
# High-frequency words:
most_important_topics = []
TopicCount = Counter(" ".join(newdset.tweet).split()).most_common(15)
#print(TopicCount)
for topic_name in TopicCount:
     most_important_topics.append(topic_name[0])
print(most_important_topics)
#pd.Series(' '.join(newdset.tweet).split()).value_counts()[:15]

#LDA
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm=cv.fit_transform(newdset['tweet'])
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

if earthquake_found:
  event_type = "earthquake"
elif fire_found:
  event_type = "Fire"
else:
     event_type="None"
     sys.exit()

#if(event_type=="None"):
#     return

print(event_type)

# Semantic Analysis
ground_truth_semantic = nlp("Earthquake")
for index, row in newdset.iterrows():
  row_nlp = nlp(row.tweet)
  print(ground_truth_semantic.similarity(row_nlp))
  if index==20:
    break

# NER
for index, row in newdset.iterrows():
  row_nlp = nlp(row.tweet)
  row_nlp2 = nlp2(row.tweet)
  if row_nlp2.ents:
      for ent in row_nlp2.ents:
           if ent.label_=="HLP":
                help_tweets[index]={"tweet": "", "location": "", "timestamp": ""}
                help_tweets[index]['tweet']=(dset.loc[index].tweet)
                help_tweets[index]['timestamp']=(dset.loc[index].time)
  if row_nlp.ents:
    for ent in row_nlp.ents:
      if ent.label_=="GPE":
        #if ent.text in state_list:
          loc_name = ent.text.lower()
          if index in help_tweets:
               help_tweets[index]['location']=loc_name
          if loc_name not in location_dictionary:
            location_dictionary[loc_name] = 0
          location_dictionary[loc_name] = location_dictionary[loc_name] + 1 
        #print(ent.text+ "\t"+ str(row.lat) + "\t" + str(row.long))    
  #if index==750:
       #break

print(location_dictionary)

most_frequent_locations = sorted(location_dictionary.items(), key = lambda kv:(kv[1], kv[0]))
most_frequent_locations = most_frequent_locations[(len(most_frequent_locations)-3):]

print (most_frequent_locations)
# Most frequent location
#most_frequent_location = max(location_dictionary, key=location_dictionary.get)

# Updating Event Dictionary
#event_list.append({event_type: most_frequent_locations})
#event_list.append({"event_type": event_type, "locations": most_frequent_locations})
event_list[event_type] = most_frequent_locations
print(event_list)

# Splitting location names and frequencies
loc_names = []
loc_freqs = []
for el in most_frequent_locations:
     loc_names.append(el[0])
     loc_freqs.append(str(el[1]))
     
    
# Parsing help tweets to be compatible for GraphQL     
help_tweet_str = ''  
for h_twt in list(help_tweets.values()):
     help_tweet_str = "{orgStr}{{tweet:\\\"{rowTweet}\\\", location:\\\"{rowLoc}\\\", timestamp: \\\"{rowTime}\\\", tweet_id: \\\"\\\", status: \\\"1\\\" }},".format(orgStr=help_tweet_str, rowTweet=h_twt['tweet'], rowLoc=h_twt['location'], rowTime=datetime.strptime(h_twt['timestamp'], "%Y-%m-%d %H:%M:%S"))
print(help_tweet_str[:-1])

#Saving data to MongoDB
import requests

url = 'https://us-east-1.aws.realm.mongodb.com/api/client/v2.0/app/neptune_realm_1-uwjwy/graphql'
headers = {
  'apiKey': 'tnpTbJUWMxGtVF5jHebxZRmjlI7Lxm45W0Gabc6diaVqs2IaVpzpm3Fg5gXzQu4A',
  'Content-Type': 'application/json'
}
#data = "\t{{\r\n\t  \t\"query\": \"mutation {{ updateOneCurrent_event( query: {{ event_type: \\\"{ev}\\\" }} set: {{  locations: [\\\"{loc2}\\\", \\\"{loc1}\\\", \\\"{loc0}\\\"], location_frequency: [\\\"{freq2}\\\", \\\"{freq1}\\\", \\\"{freq0}\\\"] }} ) {{ event_type locations location_frequency }} }}\",\r\n\t\t\"variables\": null\r\n\t}}".format(ev=event_type, loc2=loc_names[2], loc1=loc_names[1], loc0=loc_names[0], freq2=loc_freqs[2], freq1=loc_freqs[1], freq0=loc_freqs[0], hlp_twt_array=list(help_tweets.values()))
data = "\t{{\r\n\t  \t\"query\": \"mutation {{ upsertOneCurrent_event( query: {{ event_type: \\\"{ev}\\\" }} data: {{  event_type: \\\"{ev}\\\", locations: [\\\"{loc2}\\\", \\\"{loc1}\\\", \\\"{loc0}\\\"], location_frequency: [\\\"{freq2}\\\", \\\"{freq1}\\\", \\\"{freq0}\\\"] }} ) {{ event_type locations location_frequency }} }}\",\r\n\t\t\"variables\": null\r\n\t}}".format(ev=event_type, loc2=loc_names[2], loc1=loc_names[1], loc0=loc_names[0], freq2=loc_freqs[2], freq1=loc_freqs[1], freq0=loc_freqs[0], hlp_twt_array=list(help_tweets.values()))
data2 = "{{\"query\": \"mutation {{ insertManyHelp_tweets(data: [{hlp_twt_array}]){{ insertedIds}} }} \"}}".format(hlp_twt_array=help_tweet_str[:-1])

#response = requests.request(
#  'POST',
#  'https://us-east-1.aws.realm.mongodb.com/api/client/v2.0/app/neptune_realm_1-uwjwy/graphql',
#  data=data,
#  headers=headers,
#)
#
#response2 = requests.request(
# 'POST',
#  'https://us-east-1.aws.realm.mongodb.com/api/client/v2.0/app/neptune_realm_1-uwjwy/graphql',
#  data=data2,
#  headers=headers,
#)
#
#print(response.json())



