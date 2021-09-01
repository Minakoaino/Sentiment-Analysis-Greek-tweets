#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import preprocessor as preproc
from nltk.tokenize import RegexpTokenizer

df = pd.read_csv('data/final.csv')
df.head(4)

def delete_tonous(df, column_to_process, processed_column='Tweet'):
    
    if (processed_column != column_to_process):
        df[processed_column] = df[column_to_process]  # create new column

    # replace greek hyphend letters
    replacements = {processed_column: {'ά': 'α', 'έ': 'ε', 'ή': 'η', 'ί': 'ι', 'ό': 'ο', 'ύ': 'υ', 'ώ': 'ω', 'ϊ': 'ι'}}
    df.replace(replacements, regex=True, inplace=True)
    
    return (df)

df['Tweet'][6]

# Αφαιρούμε όλους τους μη ελληνικούς χαρακτήρες από τα tweets.
df['Tweet'] = df['Tweet'].str.replace(r'[a-zA-Z0-9]', '', regex=True)

# Αφαιρούμε τους τόνους καθώς και τα διαλυτικά από τα tweets. Τα διαλυτικά είναι σπάνια στην ελληνική γλώσσα, ωστόσο μια πολύ συχνή λέξη στα δεδομένα είναι η λέξη κορονοϊός η οποία συναντάται πολύ συχνά στα δεδομένα επομένως είναι απαραίτητο να αφαιρεθεί.
delete_tonous(df=df, column_to_process='Tweet', processed_column='Tweet').head(2)

# Αφαιρούμε όλα τα links από τα tweets.
remove_url = lambda x: re.sub('https\S+', '', str(x))
df['Tweet'] = df.Tweet.apply(remove_url)

# Μετατρέπουμε τα tweets σε μικρά γράμματα
to_lowercase = lambda x : x.lower()
df['Tweet'] = df.Tweet.apply(to_lowercase)
delete_tonous(df=df, column_to_process='Tweet', processed_column='Tweet').head(3)

# Αφαιρούμε όλα τα σημεία στήξης από τα δεδομένα
remove_puncs = lambda x: x.translate(str.maketrans('','',string.punctuation))
df['Tweet'] = df.Tweet.apply(remove_puncs)
df['Tweet']

# Στη στήλη Location υπάρχουν πολλές κενές εγγραφές. Τις αντικαθηστούμε με 'prefer not to say' προκειμένου να σβήσουμε στη συνέχεια τις κενές εγγραφές από τη στήλη Tweet. Aφαιρούμε τη συνέχεια τα επαναλαμβανόμενα tweets καθώς και τις κενές εγγραφές	
df["Location"].fillna('prefer not to say', inplace = True)

# Επιβεβαιώνουμε την αλλαγή
df[df['Location'] == "prefer not to say"].head()

df['Tweet'] = df.Tweet.drop_duplicates()
df = df.dropna()

# Βλέπουμε αν υπάρχουν τυχόν υπολοιπόμενες κενές εγγραφές στη στήλη Τweet
df['Tweet'].isna().sum()

# Μετράμε το μέσο μάκρος των Tweets μετά την αφαίρεση των links και των mentions.
df['length'] = df['Tweet'].apply(len)

# Bλέπουμε τα στατιστικά στοιχεία του μάκρους των tweets. Υπάρχει κάποιο tweet με 0 χαρακτήρες. Εφόσον είναι μόνο ένα, θα το σβήσουμε στη συνέχεια από το αρχείο μας.
df.length.describe()

#Average tweet
df[df['length'] == 140]['Tweet'].iloc[0]

#Longest tweet
df[df['length'] == 282]['Tweet'].iloc[0]

#tweet with 0 characters
df[df['length'] == 0]

#Βλέπουμε τη κατανομή του μήκους των tweets
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
df['length'].plot(bins=100, kind='hist');

# ### Removing Stopwords
# 
# Είναι πολύ σημαντικό να αφαιρέσουμε τις stopwords γιατί αποτελούν τις λέξεις που επαναλαμβάνονται συνεχώς και δεν δίνουν κάποιο νόημα στη φράση. Ενημερώνουμε το πακέτο της nltk με λέξεις που εντοπίστηκαν στα δεδομένα μας και θα ήταν καλό να αφαιρεθούν.

from nltk.corpus import stopwords

stop_words = stopwords.words('greek') + ['μητσοτακηπαραιτησου', 'η', 'ειναι', 'με', 'θα', 'απο', 'τα', 'ο', 'την', 'του', 'σε', 'οτι', 'της', 'τον', 'οι', 'στο', 'αν', 'τις', 'τη', 'κ', 'σας', 'νδ', 'οι', 'ο', 'η', 'μου', 'σου', 'τα', 'απο', 'βαρυμπομπη', 'μμεξεφτιλες', 'μητσοτακη', 'αδωνις', 'εμβολιο', 'χαχαχα', 'αν', 'astrazeneca', 'astra', 'zeneca', 'phizer', 'α', 'ε', 'ν', 'via', 'tι', 'marka149133376', 'ο', 'χ', 'length', 'rt', 'political', 'fediuld76', 'πιο', 'ποθενεσχες', 'frq', 'COVID19greece', 'COVID19', 'Covid19', 'Marka149133376', 'covid19gr', 'covid19greece', 'covid', 'εγω', 'εσυ', 'adonisgeorgiadi', 'κανω','αλλο', 'κανεις',  'σαν', 'κατι', 'πριν', 'ολα', 'εχουν', 'κανουν', 'εχει', 'έχουν', 'κανουν', 'οπως', 'μια', 'ενα', 'amp', 'στις', 'στα', 'στους', 'εδω', 'της', 'τους', 'μας', 'ρε', '–', 'ουτε', 'εχω', 'οταν', 'σου', 'μητσοτακηγαμιεσαι', 'μου']
remove_words = lambda x : ' '.join([word for word in x.split() if word not in stop_words])
df['Tweet_without_stopwords'] = df.Tweet.apply(remove_words)
df['Tweet_without_stopwords']

#Tokenize tweets
tokenizer = RegexpTokenizer(r'\w+')
df['Tokens'] = df['Tweet_without_stopwords'].apply(lambda text: tokenizer.tokenize(text))

#inspecting some tweets
df['Tokens'][7]

words_list = [word for line in df.Tweet_without_stopwords for word in line.split()]
words_list[:20]


# Προκειμένου να βρούμε τις πιο σηνυθισμένες λέξεις που εντοπίζονται στα tweets θα χρησιμοποιήσουμε το πακέτο collections από τη βιβλιοθήκη Counter. Μέσω της συνάρτησης most_common() μπορούμε να εντοπίσουμε τις πιο σηνυθισμένες λέξεις στα δεδομένα μας, δηλαδή στη μεταβλητή word_list που δημιουργήθηκε. Θα μετατρέψουμε αυτές τις λέξεις σε ένα νέο αρχείο δεδομένων.
word_counts = Counter(words_list).most_common(50)
word_counts

# Δημιουργούμε ένα καινούργιο dataFrame με στήλες την λέξη και τη συχνότητα της λέξης. 
words_df = pd.DataFrame(word_counts)
words_df.columns = ['word', 'frq']
words_df.head()
px.bar(words_df, x='word', y='frq', title='Οι πιο συχνές λέξεις')

#WordCloud
wordcloud = WordCloud(width = 800, height = 400, random_state = 21, max_font_size = 100, collocations = False).generate(str(df.Tweet_without_stopwords))

plt.figure(figsize = (20, 10))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off');

#Save the new file
df.to_csv('data/modified.csv')

