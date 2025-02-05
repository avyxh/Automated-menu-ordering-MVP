
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
relative_path = "large_data.csv"
full_path = os.path.join(absolute_path, relative_path)
df = pd.read_csv(full_path)
# print(df)

X_train, X_test, y_train, y_test = train_test_split(
    df['descriptor'], df['item'], shuffle=True, test_size=0.2, random_state=1)

tfidf = TfidfVectorizer(sublinear_tf=True,
                        min_df=25,
                        max_df=125,
                        norm='l1',
                        ngram_range=(1, 1),
                        stop_words='english')

X_train_counts = tfidf.fit_transform(X_train)
X_test_counts = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_counts, y_train)

y_pred_prob = model.predict_proba(X_test_counts)
y_predict_0 = y_pred_prob[:, 0]
y_predict_1 = y_pred_prob[:, 1]
predicted = pd.DataFrame()
predicted["Big Mac"] = y_predict_0
predicted["Iced Coffee"] = y_predict_1
print(predicted.head())

y_pred = model.predict(X_test_counts)
acc = metrics.accuracy_score(y_test, y_pred)
print(f"\nFinal accuracy: {acc*100}%")

# Testing with multiple categories (menu items)

df1 = pd.read_csv(os.path.join(
    os.path.dirname(__file__), "multiple_items.csv"))
df2 = pd.read_csv(os.path.join(os.path.dirname(
    __file__), "multiple_items_tester.csv"))

df_test = pd.read_csv(os.path.join(os.path.dirname(
    __file__), "Vague_dataset.csv"))


def test(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['descriptor'], df['item'], shuffle=True, test_size=0.2, random_state=None)

    tfidf = TfidfVectorizer(sublinear_tf=True,
                            min_df=25,
                            max_df=150,
                            norm='l1',
                            ngram_range=(1, 1),
                            stop_words='english')

    X_train_counts = tfidf.fit_transform(X_train)
    X_test_counts = tfidf.transform(X_test)

    model = MultinomialNB(alpha=1e-06)
    model.fit(X_train_counts, y_train)

    y_pred_prob = model.predict_proba(X_test_counts)
    y_predict_0 = y_pred_prob[:, 0]
    y_predict_1 = y_pred_prob[:, 1]
    y_predict_2 = y_pred_prob[:, 2]
    predicted = pd.DataFrame()
    predicted["Chicken McNuggets"] = y_predict_0
    predicted["Iced Coffee"] = y_predict_1
    predicted["Quarter Pounder"] = y_predict_2

    y_pred = model.predict(X_test_counts)
    acc = metrics.accuracy_score(y_test, y_pred)

    return acc, model, tfidf


def test_import(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['descriptor'].values.astype('U'), df['item'].values.astype('U'), shuffle=True, test_size=0.2, random_state=None)

    tfidf = TfidfVectorizer(sublinear_tf=True,
                            min_df=1,
                            max_df=75,
                            norm='l1',
                            ngram_range=(1, 1),
                            stop_words='english')

    X_train_counts = tfidf.fit_transform(X_train)
    X_test_counts = tfidf.transform(X_test)

    model = MultinomialNB(alpha=1e-06)
    model.fit(X_train_counts, y_train)

    y_pred = model.predict(X_test_counts)
    acc = metrics.accuracy_score(y_test, y_pred)

    return acc, model, tfidf


max = 0
for i in range(100):
    curr = test(df1)
    if curr > max:
        max = curr

print(f"\nMax accuracy for initial multiple items dataset: {max*100}%")

max = 0
for i in range(100):
    curr = test_import(df_test)
    if curr > max:
        max = curr

print(f"\nMax accuracy for vague dataset: {max*100}%")

'''def test_order (order, df):
    acc, tfidf, model = test(df)
    input = np.array([order])
    input_counts = tfidf.transform(input)
    guess = model.predict(input_counts)
    print(guess[0])

#example
test_order("May I have an iced coffee?", df1)'''

def test_reply (order, df):
    acc, tfidf, model = test(df)
    input = np.array([order])
    input_counts = tfidf.transform(input)
    y_pred_prob = model.predict_proba(input_counts)
    dict = {"Chicken McNuggets" : y_pred_prob[:, 0], 
            "Iced Coffee" : y_pred_prob[:, 1], 
            "Quarter Pounder" : y_pred_prob[:, 2]}
    largest_element = y_pred_prob.max()
    if largest_element >= (2/(len(y_pred_prob[0]))):
        reply = model.predict(input_counts)[0]
    elif largest_element >= (1/(len(y_pred_prob[0]))):
        to_ask = []
        for x in range(len(y_pred_prob[0])):
            if dict[list(dict)[x]] >= (1/(len(y_pred_prob[0]))):
                to_ask.append(list(dict)[x])
                if len(to_ask) == 1:
                    reply = "Are you trying to order a " + to_ask[0] + "? Please confirm."
                else:
                    reply = "It seems like you're trying to order one of the following items: " + ", ".join(to_ask) + ". Please confirm which item you would like to order."
    else:
        reply = "I'm sorry, I didn't quite get that! Could you please try to rephrase your order?"
    print(reply)
    return reply
