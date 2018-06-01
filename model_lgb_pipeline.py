import pandas as pd
import numpy as np
import os
from time import time
from collections import Counter
from operator import itemgetter
import warnings

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings(module='sklearn*', action='ignore', 
                        category=DeprecationWarning)

def make_relative(X):
    for i in range(65):
        X[:,i] = X[:,i] / X[:,66]
    return X

def init_lgb():
    clf = LGBMClassifier(objective='multiclass',
                         num_class=4,
                         num_leaves=127,
                         min_data_in_leaf=200,
                         n_estimators=130)
    return clf
                                                   
# Train and predict with lightGBM. The return statement is used in 
# binary_feature_importance.
def main(file):
    data = pd.read_csv(file)
    data = data.dropna()
    target = data['target']
    data = data.drop(['target'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=1000) 
    X_train, X_test = (X_train.drop(['year'], axis=1), 
                       X_test.drop(['year'], axis=1))
    pipe = Pipeline([('relative', FunctionTransformer(make_relative)),
                     ('scale', StandardScaler()), 
                     #('lgb_select', SelectFromModel(init_lgb())),
                     ('lightGBM', init_lgb())])
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    score = accuracy_score(y_test, predictions)
    joblib.dump(pipe, 'filename.pkl')
    print("Accuracy: {0:.5f}".format(score))

main(os.getcwd() + '/preprocessed/COCA/dataset.csv')