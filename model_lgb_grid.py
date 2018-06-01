import pandas as pd
import numpy as np
import os
from time import time
from collections import Counter
from operator import itemgetter
import random
import warnings

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

warnings.filterwarnings(module='sklearn*', action='ignore', 
                        category=DeprecationWarning)

random.seed(42)

def make_relative(X):
    for i in range(65):
        X[:,i] = X[:,i] / X[:,66]
    return X

def init_lgb():
    clf = LGBMClassifier(objective='multiclass',
                         num_class=4,
                         num_leaves=127,
                         min_data_in_leaf=200)
    return clf

# Perform a grid-search
def grid_search(file):

    # Define the grid that has to be searched
    param_grid = {
    'lightGBM__num_leaves':[31, 70, 127],
    'lightGBM__min_data_in_leaf':[2000, 200, 20],
    'lightGBM__max_bin':[127, 255, 511]
    }

    data = pd.read_csv(file)
    data = data.dropna()
    y = data['target']
    X = data.drop(['target', 'year'], axis=1)
    pipe = Pipeline([('relative', FunctionTransformer(make_relative)),
                     ('lgb_select', SelectFromModel(init_lgb())),
                     ('scale', StandardScaler()), 
                     ('lightGBM', init_lgb())])
    print("Performing grid search...")
    grid = GridSearchCV(pipe, param_grid, cv=10, n_jobs=-1)
    grid.fit(X, y)
    print(grid.grid_scores_)
    print(grid.cv_results_)

def grid_search_randomized(file):
    param_dist = {
    'lightGBM__learning_rate':[0.1,0.2,0.3,0.4],
    'lightGBM__n_estimators':list(range(50,200))
    }
    data = pd.read_csv(file)
    data = data.dropna()
    y = data['target']
    X = data.drop(['target', 'year'], axis=1)
    pipe = Pipeline([('lightGBM', LGBMClassifier(objective='multiclass',
                                                 num_class=4,
                                                 num_leaves=127,
                                                 min_data_in_leaf=200))])
    print("Performing grid search...")
    grid = RandomizedSearchCV(pipe, param_dist, cv=3, n_iter=30, 
                              random_state=1, n_jobs=-1)
    grid.fit(X, y)
    print(grid.grid_scores_)
    print(grid.cv_results_)

if __name__ == '__main__':
    grid_search_randomized(os.getcwd() + '/preprocessed/COCA/dataset.csv')