#!/usr/bin/env python
# coding: utf-8
#5.Test_model.py

import tweepy
import re
import pickle
from tweepy import OAuthHandler
import pandas as pd
import matplotlib.pyplot as plt
import string

pd.set_option('display.max_colwidth', 1)
pd.set_option('display.max_columns', 500)

consumer_key = 'zRLze0lH2AKxY1bkVqpXfuhDZ'
consumer_secret = 'hAbqGAiTjwvTVBdUX0MMdl7Eqcb5eQPPY5jwx4SkGj07I4hGaD'
access_token = '1397618616653209602-3LUD2QVsxS5xjznEcTQJ6545G2lUko'
access_secret = 'DALjzoSMrU71FoULD4VOYN46yCtl9c1xsKghGtQGPvNxa'

# Loading the vectorizer and classfier
with open('OneVsRestClassifier.pickle','rb') as f:
    svm_model = pickle.load(f)
    
with open('model_tfidf.pickle','rb') as f:
    tfidf = pickle.load(f)    

import csv
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
args = ['εμβολιο'];
api = tweepy.API(auth,timeout=10)

# Fetching the tweets
list_tweets = []

query = args[0]
if len(args) == 1:
    for status in tweepy.Cursor(api.search, q=query+" -filter:retweets",lang='el',result_type='recent', tweet_mode = 'extended').items(500):
        list_tweets.append(status.full_text)
        
for tweet in list_tweets:
    tweet = tweet.lower()
    tweet = re.sub('https\S+', '', tweet)
    tweet = re.sub(r'@[a-zA-Z0-9]', '', tweet)
    tweet = re.sub(r'[a-zA-Z0-9]', '', tweet)
    tweet = re.sub(r"ά","α",tweet)
    tweet = re.sub(r"έ","ε",tweet)
    tweet = re.sub(r"ή","η",tweet)
    tweet = re.sub(r"ί","ι",tweet)
    tweet = re.sub(r"ύ","υ",tweet)
    tweet = re.sub(r"ώ","ω",tweet)
    tweet = re.sub(r"ϊ","ι",tweet)  
    tweet = re.sub(r"ό","ο",tweet)
    tweet = re.sub(r"ό","ο",tweet)
    tweet = re.sub(r"#","",tweet)
    tweet = re.sub(r"!","",tweet)
    tweet = re.sub(r"«","",tweet)
    tweet = re.sub(r"»","",tweet)
    tweet = re.sub(r";","",tweet)
    tweet = re.sub(r"ϋ","",tweet)
    tweet = re.sub(r"/","",tweet)    
    tweet = re.sub(r',', ' ', tweet)
    sent = svm_model.predict(tfidf.transform([tweet]).toarray())
    print(tweet,":",sent)    
    
    # Filter based on listed items
    csvw = csv.writer(open("predicted_new2", "a"))
    csvw.writerow([status.user.screen_name,
                   # created_at is a datetime object, converting to just grab the month/day/year
                   status.created_at.strftime('%m/%d/%y'),
                   sent,
                   tweet])

df = pd.read_csv('predict.csv')

#remove punctuations from sentiment column
remove_puncs = lambda x: x.translate(str.maketrans('','',string.punctuation))
df['sentiment'] = df.sentiment.apply(remove_puncs)
df['sentiment']
df.head(10)


df['sentiment'].value_counts(normalize=True) * 100


sentiment = [len(df[df['sentiment'] == 'positive']), 
             len(df[df['sentiment'] == 'negative']), 
             len(df[df['sentiment'] == 'neutral'])]

labels = ['Positive', 'Negative', 'Neutral']
colors = ['aquamarine', 'pink', 'skyblue']

t = df['sentiment'].value_counts(normalize=True) * 100
plt.style.use('ggplot')
plt.figure(figsize = (20, 10))
plt.pie(t, labels = labels, colors = colors, autopct = '%1.1f%%')
plt.title('Ανάλυση συναισθήματος με μοντέλο πρόβλεψης');

