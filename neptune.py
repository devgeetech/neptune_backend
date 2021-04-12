import spacy
import pandas as pd
nlp = spacy.load('en_core_web_md')

event_list = []
dset = pd.read_csv('napa_earthquake_unaffected_filtered_hash_cleaned.txt', encoding='ISO-8859–1', delimiter='\t')

newdset = dset.copy(deep=True)
newdset.drop(['time',	'tweet_id',	'long',	'lat'], axis=1, inplace=True)

# Removing Stop Words
all_stopwords = nlp.Defaults.stop_words
tokenizer = nlp.tokenizer
nlp.Defaults.stop_words |= {"?",".","#","I","!",":","...",} # updating stop word list

for index, row in newdset.iterrows():
  tweet_as_tokens = tokenizer(newdset.loc[index].tweet.lower())
  tokens_without_sw= [token.text for token in tweet_as_tokens if not token.text in all_stopwords]
  newdset.loc[index].tweet = ' '.join(tokens_without_sw)
  
# High-frequency words:
pd.Series(' '.join(newdset.tweet).split()).value_counts()[:20]

#LDA
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm=cv.fit_transform(newdset['tweet'])
from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=1, random_state=42)
LDA.fit(dtm)

most_important_topics = []
for i, topic in enumerate(LDA.components_):
  print(f"The Top 15 words for topic #{i}")
  most_important_topics = [cv.get_feature_names()[index] for index in topic.argsort()[-15:]]
  print(most_important_topics)

# Assigning Event Type
event_type = ""
earthquake_found = any(elem in most_important_topics  for elem in ["earthquake", "earthquakes", "quake"]) 
fire_found = any(elem in most_important_topics  for elem in ["fire", "wildfire", "flames"])

if earthquake_found:
  event_type = "earthquake"
elif fire_found:
  event_type = "Fire"

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
for index, row in dset.iterrows():
  row_nlp = nlp(row.tweet)
  if row_nlp.ents:
    for ent in row_nlp.ents:
      if ent.label_=="GPE":
        #if ent.text in state_list:
          if ent.text not in location_dictionary:
            location_dictionary[ent.text] = 0
          location_dictionary[ent.text] = location_dictionary[ent.text] + 1 
        #print(ent.text+ "\t"+ str(row.lat) + "\t" + str(row.long))
  if index==200:
       break

# Most frequent location
most_frequent_location = max(location_dictionary, key=location_dictionary.get)
print(most_frequent_location)

# Updating Event Dictionary
event_list.append((event_type, most_frequent_location, location_dictionary[most_frequent_location]))
print(event_list)






