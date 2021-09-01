#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from nltk.tokenize import RegexpTokenizer

pd.set_option('display.max_colwidth', 1)
pd.set_option('display.max_columns', 500)

df = pd.read_csv('data/sentiment_analysis_data.csv')
df.head()
df[df['label'] == 'negative'].head()
df['label'].value_counts(normalize=True) * 100

df.dropna(inplace = True)

sns.countplot(data=df, x = 'label');

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score

tfidf = TfidfVectorizer(max_features=45000)
X = df['Tweet_without_stopwords']
y = df['label']

X = tfidf.fit_transform(df['Tweet_without_stopwords'].values.astype('U'))  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,  shuffle = True, random_state = 10)


# ### Support vector machines
svm = LinearSVC(max_iter=40000)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy_score(y_test, y_pred, normalize=True)
print(classification_report(y_test, y_pred))

sample = ["ΤΕΛΕΙΑ ΜΕΡΑ ΣΗΜΕΡΑ"]
sample = tfidf.transform(sample).toarray()
sentiment = svm.predict(sample)
print('Η πρόταση είναι',":",sentiment)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


# Creates a confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index = ['Negative','Neutral', 'Positive'], 
                     columns = ['Negative','Neutral', 'Positive'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Support Vector Classification \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

sample = ["πολυ κακες οδηγιες χρησεως, επομενως δεν γνωριζεις αν το εκανες σωστα. Κακες οδηγιες ερμηνειας του αποτελεσματος. Στα σκιτσα η εικονα διαφερει πολυ απο την πραγματικη, οποτε δεν μπορεις να εισαι σιγουρος αν ερμηνευεις σωστα το αποτελεσμα. Εμενα μου εβγαλε μια πολυ χονδρη γραμμη στο C και μια ποΛυ λεπτη και αδιορατη στο Τ (στις οδηγιες για την ερμηνεια ολες οι γραμμες των σκιτσων ειναι λεπτες.). να το παρω σαν θετικο η να το παρω ως αρνητικο? Τωρα με εβαλε σε ανησυχια και πρεπει να κανω μοριακο. Μη αξιοπιστο λοιπον"]
sample = tfidf.transform(sample).toarray()
sentiment = svm.predict(sample)
print('Η πρόταση είναι',":",sentiment)

import pickle
pickle.dump(svm, open('models/model_svm.pickle', 'wb'))
pickle.dump(tfidf, open('models/model_tfidf.pickle', 'wb'))

# ### LinearSVC with OneVsRestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

ovsrc = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)
linear_ovsrc = ovsrc.predict(X_test)
accuracy_score(y_test, linear_ovsrc, normalize=True)
print(classification_report(y_test, linear_ovsrc))


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# Creates a confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index = ['Negative','Neutral', 'Positive'], 
                     columns = ['Negative','Neutral', 'Positive'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Support Vector Classification \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, linear_ovsrc)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

sample = ["Πήρα το τεστ για το σπίτι να το εχω άμεσα διαθεσιμο. Έχω ηδη υποβληθεί σε 3 τεστ (λόγω εργασίας σε νοσοκομείο) τα οποία έδειξαν ακριβώς τα ίδια αποτελέσμα τα με τη μεθοδο RT-PCR και τα υπόλοιπα rapid test που κάνω εκεί."]
sample = tfidf.transform(sample).toarray()
sentiment = ovsrc.predict(sample)
print('Η πρόταση είναι',":",sentiment)

pickle.dump(ovsrc, open('models/OneVsRestClassifier.pickle', 'wb'))


# ### Logistic Regression Classifier
# Training the classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(multi_class='multinomial')
classifier.fit(X_train, y_train)

# Testing model performance
sent_pred_log = classifier.predict(X_test)
accuracy_score(y_test, sent_pred_log, normalize=True)
print(classification_report(y_test, sent_pred_log))

sample = ["ΤΕΛΕΙΑ ΜΕΡΑ ΣΗΜΕΡΑ"]
sample = tfidf.transform(sample).toarray()
sentiment = classifier.predict(sample)
print('Η πρόταση είναι',":",sentiment)



print(classification_report(y_test, sent_pred_log))

# Creates a confusion matrix
cm = confusion_matrix(y_test, sent_pred_log)
# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index = ['Negative','Neutral', 'Positive'], 
                     columns = ['Negative','Neutral', 'Positive'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Logistic Regression Classification \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, sent_pred_log)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# ### Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score

rfc = RandomForestClassifier(n_estimators= 200, max_depth=None).fit(X_train,y_train)
sent_pred_random_forest = rfc.predict(X_test)
accuracy_score(y_test, sent_pred_random_forest, normalize=True)
print(classification_report(y_test, sent_pred_random_forest))

# Creates a confusion matrix
cm = confusion_matrix(y_test, sent_pred_random_forest)
# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index = ['Negative','Neutral', 'Positive'], 
                     columns = ['Negative','Neutral', 'Positive'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

sample = ["Tι ΟΜΟΡΦΗ μέρα που έχει σήμερα. Θα πάω για ποδήλατο"]
sample = tfidf.transform(sample).toarray()
sentiment = rfc.predict(sample)
print('Η πρόταση είναι',":",sentiment)


# ### Random Forest with OneVsRestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

ovsrc = OneVsRestClassifier(RandomForestClassifier(random_state=0)).fit(X, y)
rf_ovsrest = ovsrc.predict(X_test)

accuracy_score(y_test, rf_ovsrest, normalize=True)
print(classification_report(y_test, rf_ovsrest))

# Creates a confusion matrix
cm = confusion_matrix(y_test, rf_ovsrest)
# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index = ['Negative','Neutral', 'Positive'], 
                     columns = ['Negative','Neutral', 'Positive'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

sample = ["Tι ΟΜΟΡΦΗ μέρα που έχει σήμερα. Θα πάω για ποδήλατο"]
sample = tfidf.transform(sample).toarray()
sentiment = ovsrc.predict(sample)
print('Η πρόταση είναι',":",sentiment)

# ### Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)

from sklearn.naive_bayes import MultinomialNB

sent_pred_bayes = clf.predict(X_test)
accuracy_score(y_test, sent_pred_bayes, normalize=True)
print(classification_report(y_test, sent_pred_bayes))

from sklearn.metrics import confusion_matrix

# Creates a confusion matrix
cm = confusion_matrix(y_test, sent_pred_bayes)
# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index = ['Negative','Neutral', 'Positive'], 
                     columns = ['Negative','Neutral', 'Positive'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Nayve Bayes \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, sent_pred_bayes)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

sample = ["τελεια μερα"]
sample = tfidf.transform(sample).toarray()
sentiment = clf.predict(sample)
print('Η πρόταση είναι',":",sentiment)

