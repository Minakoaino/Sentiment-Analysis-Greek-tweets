#!/usr/bin/env python
# coding: utf-8

# Σε αυτό το στάδιο εξερευνούμε τα δεδομένα έτσι ώστε να τα "γνωρίσουμε καλύτερα".
# Τα ερωτήματα που θα απαντηθούν από τα δεδομένα σε αυτό το στάδιο:
# * Ποιοι είναι οι χρήστες με τους πιο πολλούς ακολούθους;
# * Σε ποια περιοχή διαμένουν οι περισσότεροι χρήστες;
# * Ποιες είναι οι πιο συνηθισμένες συσκευές που επιλέγουν οι χρήστες;
# * Πως διακυμένονται τα tweets σύμφωνα με την ημερομηνία; Ποια είναι η μέρα με τις πιο πολλές δημοσιεύσεις;
# * Τι ποσοστό των tweets περιέχουν link, hashtag, mention;

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv('data/final.csv')
df.head()
print("Number of tweets: {}".format(len(df)))

#Visualize the Source column percentages.
colors = ['#93baf5','#6081b5', '#cbf7e6', '#5db0d4', '#7fa0fa']
p = df.Source.value_counts().head(5).plot.pie(x='lab', y='val', autopct='%1.1f%%', rot=2, colors=colors, figsize=(15,10));
p.set_title("Σηνυθισμένες συσκευές που χρησιμοποιούν οι Έλληνες του Twitter");

tweet_id = 222
tweet = df.iloc[tweet_id]
print("Tweet: {}".format(tweet["Tweet"]))

# Με τη συνάρτηση df.groupby().mean().nlargest() εντοπίζουμε τους χρήστες 
# με τους περισσότερους ακολούθους και τους κατατάσουμε με το όνομα χρήστη τους.

most_followers = df.groupby('Username')['Followers'].mean().nlargest(15)
most_followers

ax = most_followers.plot(kind='barh', figsize=(10, 12), color='#b4aeeb', zorder=2, width=0.85)
plt.gca().invert_yaxis()
sns.despine(bottom = True, left = True)
plt.ylabel(None)
plt.xticks(None)
plt.xticks([])
plt.yticks(fontsize=18, rotation=0)

for index, value in enumerate(most_followers):
    plt.text( value, index, str(round(value, 2)), va = 'center', ha='left', fontsize=16)
    
plt.suptitle('Οι χρήστες με τους περισσότερους followers'.title(), fontsize=20)
plt.show()

most_tweets_users = df.Username.value_counts().reset_index()
most_tweets_users.columns = ['Username','counts']
most_tweets_users.head(20)

users = df['Username'].apply(pd.Series).stack().value_counts().head(10)
ax = users.plot(kind='barh', figsize=(10, 12), zorder=2, width=0.85)
plt.gca().invert_yaxis()
sns.despine(bottom = True, left = True)
plt.ylabel(None)
plt.xticks(None)
plt.xticks([])
plt.yticks(fontsize=18, rotation=0)

for index, value in enumerate(users):
    plt.text( value, index, str(round(value, 2)), va = 'center', ha='left', fontsize=16)
    
plt.suptitle('Οι χρήστες που εμφανίζονται πιο πολύ στο αρχείο δεδομένων'.title(), fontsize=15)
plt.show()


# Βρίσκουμε τη περιοχή από την οποία δημοσιεύτηκαν τα περισσότερα tweets.
most_tweets_users = df.Location.value_counts().reset_index()
most_tweets_users.columns = ['Location','counts']
most_tweets_users.head(20)

colors = ['#93baf5','#d5abd9', '#cbf7e6', '#5db0d4', '#ebcaea','#a2ebe1','#a9aade']

p = df.Location.value_counts().head(7).plot.pie(x='Location', autopct='%1.1f%%', startangle=90, 
                                               rot=2, colors=colors, figsize=(14,10));
p.set_title("Location pie chart");
plt.ylabel(None);

date_counts = df[['Tweet', 'Date']].groupby(['Date']).count().reset_index()
most_tweets = date_counts.groupby('Date')['Tweet'].mean().nlargest(20).reset_index()
most_tweets.columns = ['date','count']
most_tweets.head(20)

import plotly.express as px
px.line(most_tweets, x = 'date', y = 'count', title = 'Tweet counts per day lineplot')


# Υπολογίζουμε τα Tweets που περιέχουν hashtag, το οποίο συμβολίζεται με '#'.
tweets_with_hashtag = df[df['Tweet'].str.contains('#')==True]
print("Ο αριθμός των tweets που περιέχουν hashtag: {}".format(len(tweets_with_hashtag)))


# Με ανάλογο τρόπο μπορούμε να εντοπίσουμε και τα tweets που δεν περιέχουν hashtag σε μια συλλογή με tweets.
tweets_without_hashtag = df[df['Tweet'].str.contains('#')==False]
print("Ο αριθμός των tweets που δεν περιέχουν hashtag: {}".format(len(tweets_without_hashtag)))

#Save the file
hastag_frame = pd.read_csv('HASHTAG.csv')

hashtag = hastag_frame.has_hashtag.value_counts().plot.pie(x='lab', y='val', autopct='%1.1f%%', rot=2,figsize=(12,8));
plt.suptitle('Ποσοστό δημοσιεύσεων με hashtag'.title(), fontsize=20);


# Υπολογίζουμε τα tweets που περιέχουν link και στη συνέχεια βλέπουμε με την εντολή sum() τον αριθμό τους.
tweets_with_url = df[df['Tweet'].str.contains('http')==True]
print("Ο αριθμός των tweets που περιέχουν link: {}".format(len(tweets_with_url)))

# Επιβεβαιώνουμε την ύπαρξη link σε κάποιες από τις εγγραφές
tweets_with_url['Tweet'][37756]
# Υπολογίζουμε τα tweets που δεν περιέχουν link και στη συνέχεια βλέπουμε με την εντολή sum() τον αριθμό τους.
tweets_without_url = df[df['Tweet'].str.contains('http')==False]
print("Ο αριθμός των tweets που δεν περιέχουν link: {}".format(len(tweets_without_url)))

url_frame = pd.read_csv('url_frame.csv')
tag = url_frame.has_url.value_counts().head(5).plot.pie(x='lab', y='val', autopct='%1.1f%%', rot=2,figsize=(12,8));
plt.suptitle('Ποσοστό δημοσιεύσεων με link'.title(), fontsize=20);


# Υπολογίζουμε τα tweets που περιέχουν αναφορά (mention) η οποία στο Twitter συμβολίζεται με @.
tweets_with_mention = df[df['Tweet'].str.contains('@')== True]
print("Ο αριθμός των tweets που περιέχουν mention: {}".format(len(tweets_with_mention)))

tweets_without_mention = df[df['Tweet'].str.contains('@')== False]
print("Ο αριθμός των tweets που δεν περιέχουν mention: {}".format(len(tweets_without_mention)))

mention_frame = pd.read_csv('MENTIONS.csv')

tag = mention_frame.has_mention.value_counts().head(5).plot.pie(x='lab', y='val', autopct='%1.1f%%', rot=2,figsize=(12,8));
plt.suptitle('Ποσοστό δημοσιεύσεων με mention'.title(), fontsize=20);

tweets_with_emojis = df[df['Tweet'].str.contains('😂')== True]
print("Ο αριθμός των tweets που περιέχουν το emoji: {}".format(len(tweets_with_emojis)))

tweets_with_emojis['Tweet'][198]

# Υπολογίζουμε τα tweets που είναι retweets και όχι αυτούσια. Στο twitter αυτό συμβολίζεται με RT

retweets = df[df['Tweet'].str.startswith('RT')== True]
print("Ο αριθμός των retweets: {}".format(len(retweets)))

retweets = pd.read_csv('Retweeted.csv')

no_retweets = df[df['Tweet'].str.startswith('RT')== False]
print("Ο αριθμός των tweets που δεν είναι retweet: {}".format(len(no_retweets)))

retweets_plot = retweets.is_retweet.value_counts().head().plot.pie(x='lab', y='val', autopct='%1.1f%%', rot=2,figsize=(12,8));
plt.suptitle('Ποσοστό δημοσιεύσεων που είναι retweets'.title(), fontsize=20);

