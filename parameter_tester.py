from time import time
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
from sklearn.pipeline import Pipeline
from pprint import pprint
from sklearn.model_selection import GridSearchCV

absolute_path = os.path.dirname(__file__)
relative_path = "three_items_import.csv"
full_path = os.path.join(absolute_path, relative_path)
df = pd.read_csv(full_path)

X_train, X_test, y_train, y_test = train_test_split(
    df['descriptor'], df['item'], shuffle=True, test_size=0.2, random_state=None)

# all the code following this is to test various parameters for the vectorizer and classifier
# and identify ideal parameters
pipeline = Pipeline([("vect", TfidfVectorizer()), ("clf", MultinomialNB())])

parameter_grid = {
    "vect__max_df": (75, 100, 125, 150, 175, 200),
    "vect__min_df": (5, 7, 9, 10, 25, 50, 75),
    "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
    "vect__norm": ("l1", "l2"),
    "clf__alpha": np.logspace(-6, 6, 13),
}


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    n_jobs=2,
    verbose=1,
)

print("Performing grid search...")
print("Hyperparameters to be evaluated:")
pprint(parameter_grid)


t0 = time()
grid_search.fit(X_train, y_train)
print(f"Done in {time() - t0:.3f}s")

print("Best parameters combination found:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameter_grid.keys()):
    print(f"{param_name}: {best_parameters[param_name]}")

test_accuracy = grid_search.score(X_test, y_test)
print(
    "Accuracy of the best parameters using the inner CV of "
    f"the random search: {grid_search.best_score_:.3f}"
)
print(f"Accuracy on test set: {test_accuracy:.3f}")
