import pandas as pd
import numpy as np
import os
from time import time
from collections import Counter
from operator import itemgetter

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Parameters for the lightGBM model.
params = {
    'objective': 'multiclass',
    'num_class': 4,
    'num_leaves': 127,
    'learning_rate': 0.2,
    'min_data_in_leaf':200,
}

# Return a list with the features ranked from most to least important. A 
# feature is important when it is used a lot in the model (this is an 
# actual number, like '345')
def get_important_features(clf):
    importances = clf.feature_importance()
    features = clf.feature_name()
    n_features = len(features)
    pairs = [(importances[i], features[i]) for i in range(n_features)]
    pairs.sort(key=itemgetter(0), reverse=True)
    return [pair[1] for pair in pairs]

# Shows which which genres are confused with one another
def wrong_preds(y_pred, y_test):
    N = len(y_pred)
    wrong = []
    for i in range(N):
        if y_pred[i] != y_test[i]:
            wrong.append((y_test[i], y_pred[i]))
    return Counter(pred for pred in wrong)

# Get some info about the predictions/model
def get_stats(clf):
    lgb.plot_importance(clf, max_num_features=20)
    plt.show()
    print(get_important_features(clf))
    return 0

def get_distribution(year, target):
    data = pd.concat([year, target], axis=1)
    years = list(range(1990, 2016)) # sorted(list(set(year)))
    targets = [0,1,2,3] # sorted(list(set(target)))
    genres = ['academic', 'fiction', 'newspaper', 'magazine']
    df = pd.DataFrame(index=years)
    for i in targets:
        result = []
        for year in years:
            instances = data[(data['year'] == year) & (data['target'] == i)]
            result.append(len(instances))
        df[genres[i]] = result
    df.plot(kind='bar', stacked=True)
    plt.xlabel('Year')
    plt.ylabel('Number of instances')
    plt.title('Instance distribution sorted by genre and year')
    plt.show()
    return 0

# Remove the 'year' feature from the test and training set. 
def remove_year(X_train, X_test):
    return X_train.drop(['year'], axis=1), X_test.drop(['year'], axis=1)

# Get the accuracy of the predictions made by the model
def get_score(clf, y_test, X_test):
    y_pred = clf.predict(X_test)
    y_pred = [np.argmax(line) for line in y_pred]
    score = accuracy_score(y_test, y_pred)
    print("Score: {0:.2f}".format(score))
    return 0

# Train and predict with lightGBM. The return statement is used in 
# binary_feature_importance.
def main(file, binary=None):
    data = pd.read_csv(file)
    data = data.dropna()
    target = data['target']
    if binary != None:
        target = target.apply(lambda x:0 if x != binary else 1)
        params['objective'] = 'binary'
        params['num_class'] = 1
    data = data.drop(['target'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, target) 
    get_distribution(X_train['year'], y_train)
    X_train, X_test = remove_year(X_train, X_test)
    print(X_train.head())
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    clf = lgb.train(params, lgb_train)
    clf.save_model('lgb_model.txt')
    get_score(clf, y_test, X_test)
    # get_stats(clf)
    return get_important_features(clf)

# Train and predict with lightGBM. The return statement is used in 
# binary_feature_importance.
def lgb_classifier(file, binary=None):
    data = pd.read_csv(file)
    data = data.dropna()
    target = data['target']
    data = data.drop(['target'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, target) 
    X_train, X_test = remove_year(X_train, X_test)
    clf = LGBMClassifier(objective='multiclass',
                     num_leaves=127,
                     learning_rate=0.2,
                     min_data_in_leaf=200,
                     num_class=4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)

# Train and predict with lightGBM. The return statement is used in 
# binary_feature_importance.
def cros_val(file):
    data = pd.read_csv(file)
    data = data.dropna()
    y = data['target']
    X = data.drop(['target', 'year'], axis=1)
    clf = LGBMClassifier(objective='multiclass',
                         num_class=4,

                         n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=10)
    print(scores)
    print(scores.mean())
    print(scores.std())

# Every iteration singles out a genre X, and trains the model on X or not-X. 
# Afterwards, the most important features are saved. 
def binary_feature_importance():
    df = pd.DataFrame()
    genres = ['academic', 'fiction', 'magazine', 'newspaper']
    for i in range(4):
        features = main(os.getcwd() + "/preprocessed/COCA/" +
                        "all_features.csv", params, i)
        df[genres[i]] = features
    df.to_csv('binary_feature_importance.csv')
    return 0

if __name__ == '__main__':
    cros_val(os.getcwd() + '/preprocessed/COCA/dataset.csv')
