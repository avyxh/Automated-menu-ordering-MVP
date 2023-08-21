
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB


df1 = pd.read_csv("large-three-item.csv")


def test(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['descriptor'], df['item'], shuffle=True, test_size=0.2, random_state=None)

    tfidf = TfidfVectorizer(sublinear_tf=True,
                            min_df=1,
                            max_df=50,
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

    return acc, tfidf, model


def test_order(order, df):
    acc, tfidf, model = test(df)
    input_arr = np.array([order])
    input_counts = tfidf.transform(input_arr)
    guess = model.predict(input_counts)
    return guess[0]


def test_reply(order, acc, tfidf, model, df):
    input_arr = np.array([order])
    input_counts = tfidf.transform(input_arr)
    y_pred_prob = model.predict_proba(input_counts)
    dict = {"Chicken McNuggets": y_pred_prob[:, 0],
            "Iced Coffee": y_pred_prob[:, 1],
            "Quarter Pounder": y_pred_prob[:, 2]}
    largest_element = y_pred_prob.max()
    if largest_element >= (2/(len(y_pred_prob[0]))):
        return model.predict(input_counts)[0]
    elif largest_element >= (1/(len(y_pred_prob[0]))):
        to_ask = []
        for x in range(len(y_pred_prob[0])):
            if dict[list(dict)[x]] >= (1/(len(y_pred_prob[0]))):
                to_ask.append(list(dict)[x])
                if len(to_ask) == 1:
                    value = to_ask[0]
                    answer = input(
                        "It looks like you're trying to order " + value + ". Please confirm. Y / N?\n")
                    if answer.replace(" ", "").lower() == "y" or answer.replace(" ", "").lower() == "yes":
                        return value
                    else:
                        reply_rep = "Please try to input the item again.\n"
                        a = input(reply_rep)
                        item = test_order(a, df)
                        ans = input(
                            "It looks like you're trying to order " + item + ". Please confirm. Y / N?\n")
                        while (ans.replace(" ", "").lower() == "n" or ans.replace(" ", "").lower() == "no"):
                            reply_rep = "Please try to input the item again.\n"
                            item = test_order(input(reply_rep), df)
                            ans = input(
                                "It looks like you're trying to order " + item + ". Please confirm. Y / N?\n")
                        return item
                else:
                    number_list = ""
                    for x in range(len(to_ask)):
                        number_list += " " + str(x)
                    reply = "It seems like you're trying to order one of the following items: " + \
                        ", ".join(
                            to_ask) + ". Please confirm which item you would like to order by selecting the matching number" + number_list
                    menu_no = int(input(reply))
                    return to_ask[menu_no]
    else:
        reply = "I'm sorry, I didn't quite get that! Could you please try to rephrase your order?"
        a = input(reply)
        item = test_order(a, df)
        ans = input(
            "It looks like you're trying to order " + item + ". Please confirm. Y / N?\n")
        while (ans.replace(" ", "").lower() == "n" or ans.replace(" ", "").lower() == "no"):
            reply = "I'm sorry, I didn't quite get that! Could you please try to rephrase your order?\n"
            item = test_order(input(reply), df)
            ans = input(
                "It looks like you're trying to order " + item + ". Please confirm. Y / N?\n")
        return item


def run_list(my_list, df):
    items = ["Chicken McNuggets", "Iced Coffee", "Quarter Pounder"]
    acc1, tfidf1, model1 = test(df)
    current_list = []
    for x in my_list:
        item = test_reply(x, acc1, tfidf1, model1)
        if item in items:
            current_list.append(item)
        else:
            ans = test_reply(item, acc1, tfidf1, model1)
            if ans in items:
                current_list.append(ans)
            else:
                print(
                    "We're having an issue understanding this order. Moving to next item...")
    print(current_list)
    # may generate error
    order_conf = input("Please confirm if this is your final order. Y / N")
    if order_conf.replace(" ", "").lower() == "y" or order_conf.replace(" ", "").lower() == "yes":
        final_list = current_list
        return final_list
    elif order_conf.replace(" ", "").lower() == "n" or order_conf.replace(" ", "").lower() == "no":
        add_rem = input("Would you like to add or delete items? A / D")
        if add_rem.replace(" ", "").lower() == "a" or add_rem.replace(" ", "").lower() == "add":
            # needs to somehow run data through ChatGPT and obtain new list as new_list
            for x in new_list:
                item = test_reply(x, acc1, tfidf1, model1)
                if item in items:
                    current_list.append(item)
                else:
                    ans = test_reply(item, acc1, tfidf1, model1)
                    if ans in items:
                        current_list.append(ans)
                    else:
                        print(
                            "We're having an issue understanding this order. Moving to next item...")
            final_list = current_list
            return final_list
        elif add_rem.replace(" ", "").lower() == "d" or add_rem.replace(" ", "").lower() == "delete":
            for c in current_list:
                print(c + " " + str(current_list.index(c)))
            ans = input(
                "Please input the number(s) corresponding to the item(s) you would like to delete, seperated by commas")
            del_list = ans.split(",")
            for y in del_list:
                current_list.remove(y)
            final_list = current_list
            return final_list


df1 = pd.read_csv("large-three-item.csv")
acct, tfidft, modelt = test(df1)
descriptors = df1['descriptor'].tolist()
for x in range(len(descriptors)):
    test_reply(descriptors[x], acct, tfidft, modelt, df1)
