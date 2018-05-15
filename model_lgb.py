import pandas as pd
import numpy as np
import os
from time import time
from collections import Counter
from operator import itemgetter

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

params = {
    'objective': 'multiclass',
    'num_class': 5,
    'num_leaves': 31,
    'learning_rate': 0.2,
}

def get_important_features(clf):
    importances = clf.feature_importance()
    features = clf.feature_name()
    n_features = len(features)
    pairs = [(importances[i], features[i]) for i in range(n_features)]
    pairs.sort(key=itemgetter(0), reverse=True)
    return [pair[1] for pair in pairs]

def wrong_preds(y_pred, y_test):
    N = len(y_pred)
    wrong = []
    for i in range(N):
        if y_pred[i] != y_test[i]:
            wrong.append((y_test[i], y_pred[i]))
    return Counter(pred for pred in wrong)

def get_stats(clf):
    lgb.plot_importance(clf, max_num_features=20)
    plt.show()
    print(get_important_features(clf))

def main(file, params, binary=None):
    data = pd.read_csv(file)
    data = data.dropna()
    target = data['target']
    if binary != None:
        target = target.apply(lambda x:0 if x != binary else 1)
        params['objective'] = 'binary'
        params['num_class'] = 1
    data = data.drop(['target', 'year'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, target) 
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    clf = lgb.train(params, lgb_train)
    clf.save_model('lgb_model.txt')
    y_pred = clf.predict(X_test)
    y_pred = [np.argmax(line) for line in y_pred]
    get_stats(clf)
    score = accuracy_score(y_test, y_pred)
    print("Score: {0:.2f}".format(score))
    return get_important_features(clf)

# Every iteration singles out a genre X, and trains the model on X or not-X. 
# Afterwards, the most important features are saved. 
def binary_feature_importance():
    df = pd.DataFrame()
    genres = ['academic', 'fiction', 'magazine', 'newspaper']
    for i in range(4):
        features = main(os.getcwd() + "/preprocessed/COCA/" +
                        "POS_entities_relative_sentiment_readscore.csv", params, i)
        df[genres[i]] = features
    df.to_csv('binary_feature_importance.csv')
    return 0

main(os.getcwd() + '/preprocessed/COCA/POS_entities_relative_sentiment_readscore.csv', params)

