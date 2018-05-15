# Basic libraries
import numpy as np
import pandas as pd
from time import time
import os

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Initialize data. Then split it in a train and a test set, and train the model
def main(file):
    t0 = time()
    print("Loading data...")
    data = pd.read_csv(file)
    data = data.dropna()
    target = data['target']
    data = data.drop(['target'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, target) 
    print("Done, {0:.1f}s".format(time()-t0))
    t0 = time()
    print("Training...")
    pipeline = Pipeline([('kBest', SelectKBest(chi2, k='all')),
                         ('tree', RandomForestClassifier())])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print("Done, {0:.1f}s".format(time()-t0))
    score = accuracy_score(y_test, predictions)
    print("Accuracy: {0:.2f}".format(score))
    return 0

main(os.getcwd() + "/preprocessed/COCA/POS_rel.csv")