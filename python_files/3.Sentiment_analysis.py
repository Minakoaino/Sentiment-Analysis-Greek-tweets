#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import seaborn as sns
import string
import plotly.express as px

#Load the modified dataset with the translations
tweets_df = pd.read_csv('modified_with_length.csv')
tweets_df.head()

#Make english text lowercase
to_lowercase = lambda x : str(x).lower()
tweets_df['Translated'] = tweets_df.Translated.apply(to_lowercase)

#Remove english punctuations
remove_puncs = lambda x: x.translate(str.maketrans('','',string.punctuation))
tweets_df['Translated'] = tweets_df.Translated.apply(remove_puncs)
tweets_df['Translated']


# ### Συναισθηματική ανάλυση με τη μέθοδο TextBlob
#Make a copy of our data
textblob_analysis = tweets_df.copy()

#Create polarity with textblob column
textblob_analysis["Polarity"] = textblob_analysis["Translated"].apply(lambda word: TextBlob(str(word)).sentiment.polarity)
labelize_textblob = lambda x : 'neutral' if x==0 else('positive' if x>0 else 'negative')
textblob_analysis['label'] = textblob_analysis.Polarity.apply(labelize_textblob)
textblob_analysis.head()
textblob_analysis.Polarity.describe()

#drop english column
textblob_analysis.drop('Translated', axis=1, inplace=True)

# Τα πιο θετικά και πιο αρνητικά tweets που εντόπησε η μέθοδος.
textblob_analysis[textblob_analysis.Polarity == -1.000000]
textblob_analysis[textblob_analysis.Polarity == 1.000000]
textblob_analysis.describe()

#Save dataframe
# textblob_analysis.to_csv('textblob_label.csv')

# ### Προσέγγιση Vader
#Make a copy of our data
vader_analysis = tweets_df.copy()

sid = SentimentIntensityAnalyzer()
ps = lambda x : sid.polarity_scores(str(x))
vader_scores = vader_analysis.Translated.apply(ps)
vader_scores

#Make it a dataframe
sentiment_df_vader = pd.DataFrame(data = list(vader_scores))
sentiment_df_vader.head()

labelize = lambda x : 'neutral' if x==0 else('positive' if x>0 else 'negative')
sentiment_df_vader['label'] = sentiment_df_vader.compound.apply(labelize)
sentiment_df_vader.head()

#Join dataframes
vader_data_with_compound = tweets_df.join(sentiment_df_vader.compound)
vader_data_with_compound.head()

vader_data_with_label = vader_data_with_compound.join(sentiment_df_vader.label)
vader_data_with_label.head(20)

#Drop english column
vader_data_with_label.drop('Translated', axis=1, inplace=True)

# Καλώντας τη συνάρτηση describe() μπορούμε να πάρουμε μερικά στατιστικά στοιχεία για τη πολικότητα των tweets.  Βρήσκοντας τα tweets με τη πιο μικρή και πιο μεγάλη πολυκότητα μπορούμε να βρούμε το πιο αρνητικό και το πιο θετικό tweet.
vader_data_with_label.compound.describe()

# Βλέπουμε ότι το πιο θετικό tweet έχει χαρακτηριστεί με τη κλάση θετικό λόγω των πολλών emojis τα οποία περιέχει. Αυτό αποτελεί ψευδώς θετικό tweet αφού στην ουσία περιέχει ειρωνία.
vader_data_with_label[vader_data_with_label.compound == 0.997000]
vader_data_with_label['Tweet_without_stopwords'][1788]
vader_data_with_label[vader_data_with_label.compound == -0.984700]
vader_data_with_label['Tweet_without_stopwords'][3353]

#Save vader data file
vader_data_with_label.to_csv('vader_with_label.csv')

# ### Οπτικοποίηση δεδομένων
vader_pie = [len(vader_data_with_label[vader_data_with_label['label'] == 'positive']), 
             len(vader_data_with_label[vader_data_with_label['label'] == 'negative']), 
             len(vader_data_with_label[vader_data_with_label['label'] == 'neutral'])]

blob_pie = [len(textblob_analysis[textblob_analysis['label'] == 'positive']), 
            len(textblob_analysis[textblob_analysis['label'] == 'negative']), 
            len(textblob_analysis[textblob_analysis['label'] == 'neutral'])]
labels = ['Positive', 'Negative', 'Neutral']
colors = ['aquamarine', 'tomato', 'skyblue']


# Ανάλυση ποσοστού tweets ανά κατηγορία συναισθήματος με δυο διαφορετικές προσεγγίσεις. Βλέπουμε ότι οι δυο προσεγγίσεις που ακολουθήσαμε έχουν σχεδόν το ίδιο ποσοστό θετικών δημοσιεύσεων οστώσο υπάρχει μεγάλη διαφοριοποίηση στα αρνητικά αποτελέσματα.

plt.style.use('ggplot')
plt.figure(figsize = (20, 10))
plt.subplot(1, 2, 1)
plt.pie(vader_pie, labels = labels, colors = colors, autopct = '%1.1f%%')
plt.title('Ανάλυση συναισθήματος με Vader')
plt.subplot(1, 2, 2)
plt.pie(blob_pie, labels = labels, colors = colors, autopct = '%1.1f%%')
plt.title('Ανάλυση συναισθήματος με TextBlob');

#Count the labels
counts_df_vader = vader_data_with_label.label.value_counts().reset_index()
counts_df_vader
counts_df_textblob = textblob_analysis.label.value_counts().reset_index()
counts_df_textblob

# Βλέπουμε τη διακίμανση του συναισθήματος των tweets καταμετρόντας τα σύμφωνα με την ημερομηνία και τη κλάση τους και για τις δυο προσεγγίσεις.

#Vader counts
data_agg_vader = vader_data_with_label[['Tweet', 'Date', 'label']].groupby(['Date', 'label']).count().reset_index()
data_agg_vader.columns = ['date','label','counts']
data_agg_vader.head()

#Vader line visualisation
px.line(data_agg_vader, x = 'date', y = 'counts', color = 'label', title = 'Daily tweets sentimental Analysis with Vader method')
data_agg_textblob = textblob_analysis[['Tweet', 'Date', 'label']].groupby(['Date', 'label']).count().reset_index()
data_agg_textblob.columns = ['date','label','counts']
data_agg_textblob.head()
px.line(data_agg_textblob, x = 'date', y = 'counts', color = 'label', title = 'daily tweets sentimental Analysis with Textblob')

