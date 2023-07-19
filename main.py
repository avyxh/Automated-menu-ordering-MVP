
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

absolute_path = os.path.dirname(__file__)
relative_path = "test_data.csv"
full_path = os.path.join(absolute_path, relative_path)
df = pd.read_csv(full_path)
# print(df)

X_train, X_test, y_train, y_test = train_test_split(
    df['descriptor'], df['item'], shuffle=True, test_size=0.2, random_state=1)

tfidf = TfidfVectorizer(sublinear_tf=True,
                        min_df=5,
                        norm='l2',
                        ngram_range=(1, 2),
                        stop_words='english')

X_train_counts = tfidf.fit_transform(X_train)
X_test_counts = tfidf.transform(X_test)


clf = MultinomialNB()
clf.fit(X_train_counts, y_train)


y_pred = clf.predict(X_test_counts)
acc = metrics.accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc*100}%")


'''
Code for tfidf with logistic regression (gives error)

X_train, X_test, y_train, y_test = train_test_split(
    df, df['item'], shuffle=True, test_size=0.2, random_state=1)

tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)

print(X_train_vectors_tfidf)
print(y_train)

lr_tfidf = LogisticRegression(solver='liblinear', C=10, penalty='l2')
lr_tfidf.fit(X_train_vectors_tfidf, y_train)
# Predict y value for test dataset
y_predict = lr_tfidf.predict(X_test_vectors_tfidf)
y_prob = lr_tfidf.predict_proba(X_test_vectors_tfidf)[:, 1]
print(classification_report(y_test, y_predict))
print('Confusion Matrix:', confusion_matrix(y_test, y_predict))

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)
'''

'''
Code for random forest classifier in case we decide to use it


X_train, X_test, y_train, y_test = train_test_split(
    df, df['item'], shuffle=True, test_size=0.2, random_state=1)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)

print("Accuracy: " + acc)
'''
