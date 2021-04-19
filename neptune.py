import spacy
import pandas as pd
import re
from collections import Counter
nlp = spacy.load('en_core_web_md')

event_list = []
#dset = pd.read_csv('napa_earthquake_unaffected_filtered_hash_cleaned.txt', encoding='ISO-8859–1', delimiter='\t')
#dset = pd.read_csv('napa_earthquake_complete.txt', encoding='ISO-8859–1', delimiter='\t')

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
print(TopicCount)
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

# Setting up dictionaries
#state_list = ['California', 'Texas', 'New Jersey', 'Nevada']
location_dictionary = {}

# NER
for index, row in newdset.iterrows():
  row_nlp = nlp(row.tweet)
  if row_nlp.ents:
    for ent in row_nlp.ents:
      if ent.label_=="GPE":
        #if ent.text in state_list:
          loc_name = ent.text.lower()  
          if loc_name not in location_dictionary:
            location_dictionary[loc_name] = 0
          location_dictionary[loc_name] = location_dictionary[loc_name] + 1 
        #print(ent.text+ "\t"+ str(row.lat) + "\t" + str(row.long))
  if index==750:
       break

print(location_dictionary)

most_frequent_locations = sorted(location_dictionary.items(), key = lambda kv:(kv[1], kv[0]))
most_frequent_locations = most_frequent_locations[(len(most_frequent_locations)-3):]
print(most_frequent_locations)

# Most frequent location
#most_frequent_location = max(location_dictionary, key=location_dictionary.get)
#print(most_frequent_location)

# Updating Event Dictionary
event_list.append({event_type: most_frequent_locations})
print(event_list)






