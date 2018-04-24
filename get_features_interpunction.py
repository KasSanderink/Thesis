# Turn raw text into features using common sense, that is, non-statistical 
# methods. Saves new features into a csv file.
###############################################################################

# Basic libraries
import numpy as np
import pandas as pd
import os

# Import self-written scripts
from generate_documents import document_generator

# Using a generator, construct features. For now, the features are just the 
# number of occurences of some interpunction.
def generate_features():
    n_features = 8
    gen = document_generator()
    punctuation = ['.', ':', ';', '!', '?']
    n_instances = sum(1 for i in gen)
    result = np.empty([n_instances, n_features], dtype=int)
    print("Getting ready..")
    for file in gen:
        index = file[3]
        for i in range(5):
            result[index, i] = file[0].count(punctuation[i])
        result[index, 5] = len(file[0])
        result[index, 6] = file[2]
        result[index, 7] = file[1]
    print("Done")
    print("Saving...")
    np.savetxt("punctuation.csv", result, delimiter=",", fmt='%i')
    print("Done")

# Load data from csv into a dataframe
def load_df(relative=False):
    if relative:
        path = os.getcwd()+'\preprocessed\COCA\interpunction_relative.csv'
    else:
        path = os.getcwd()+'\preprocessed\COCA\interpunction_all.csv'
    df = pd.read_csv(path)
    df.columns = ['full_stop', 'colon', 'semi_colon', 'exclamation',
                 'question', 'n_chars', 'year', 'target']
    return df

# Make all interpunction features relative to the number of chars in a 
# document.
def generate_relative_interpunction():
    df = load_df()
    features = ['full_stop', 'colon', 'semi_colon', 'exclamation', 'question']
    n_chars = df['n_chars']
    for feature in features:
        df[feature] = (df[feature]/n_chars)*1000
    df.to_csv("relative_interpunction.csv", header=False, index=False)
    return 0

# Select all instances within a certain time period (between start_year and 
# star_year + step_size)
def _create_period_csv(df, start_year, step_size):
    return df[(df['year'] >= start_year) &
              (df['year'] < start_year + step_size)]

# Split up a csv file into seperate files based on publishing year
def generate_period_features(relative=False):
    first_year = 1990
    last_year = 2015
    step_size = 5
    df = load_df(relative)
    for year in range(first_year, last_year, step_size):
        period_df = _create_period_csv(df, year, step_size)
        period = str(year) + "-" + str(year+step_size)
        if relative:
            period_df.to_csv("interpunction_relative" + period + ".csv",
                         header=False, index=False)
        else:
            period_df.to_csv("interpunction" + period + ".csv",
                             header=False, index=False)
    return 0

