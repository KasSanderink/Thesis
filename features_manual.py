# Functions that take a generator as input, and give a column of features as
# output
###############################################################################

import numpy as np
import pandas as pd
import os
import warnings
import re
from time import time

import pyphen
import nltk
from nltk.tokenize import RegexpTokenizer

from generate_documents import document_generator

# readability_score() generates a warning but it's okay
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Get the average word length of a document
def get_word_length():
    gen = document_generator()
    instances = sum(1 for i in gen)
    gen = document_generator()
    result = np.empty([instances, 1])
    for file in gen:
        if words:
            n_words = words[file[i]]
        else:
            n_words = len(file[0].split())
        try:
            result[file[3]] = len(file[0]) / n_words
        except ZeroDivisionError:
            result[file[3]] = 0
    return result

# Count APA refs based on regex. Could be replaced with entity tagging.
def count_APA():
    gen = document_generator()
    n_instances = sum(1 for i in gen)
    print(n_instances)
    gen = document_generator()
    n_features = 3 + 2
    result = np.empty([n_instances, n_features])
    for file in gen:
        apa_1 = len(re.findall("\( [A-Z][a-z]* , \d\d\d\d \)", file[0]))
        et_al = len(re.findall("et\.* al",  file[0]))                     
        year = len(re.findall(" \d\d\d\d ", file[0]))
        instance = [apa_1, apa_2, year, file[2], file[1]]
        result[file[3]] = instance
    np.savetxt("apa.csv", result, delimiter=",", fmt='%i')

# You need a csv file with certain nltk tags in it to run this function
def n_POS(tags, name):
    path = os.getcwd()
    data = pd.read_csv(path+'/preprocessed/COCA/POS.csv')
    verbs = data[tags]
    result = verbs.apply(sum, axis=1)
    result.columns = name
    result.to_csv(name + '.csv', index=False, header=[name])
    return 0

def n_count():
    gen = document_generator()
    n_features = 3 + 2
    n_instances = 17821 #sum(1 for i in gen)
    result = np.empty([n_instances, n_features])
    gen = document_generator()
    dic = pyphen.Pyphen(lang='en')
    tok = RegexpTokenizer('\w+')
    for file in gen:
        raw_text = file[0]
        sent_count = len(nltk.sent_tokenize(raw_text))
        words = tok.tokenize(raw_text)
        word_count = len(words)
        syllables = [dic.inserted(word) for word in words]
        syllable_count = [len(re.findall('-', word)) + 1 for word in syllables] 
        syllable_count = sum(syllable_count)
        final = [sent_count, word_count, syllable_count, file[2], file[1]]
        result[file[3]] = final
    np.savetxt("count.csv", result, delimiter=",", fmt='%i')
    return 0

def readability_score():
    path = os.getcwd() + '/preprocessed/COCA/sent_word_syl_count.csv'
    count = pd.read_csv(path)
    arr = count.as_matrix()
    flesh = 206.835-1.015*(arr[:,1]/arr[:,0])-84.6*(arr[:,2]/arr[:,1])
    FK = 0.39*(arr[:,1]/arr[:,0])+11.8*(arr[:,2]/arr[:,1])-15.59
    df = pd.DataFrame()
    df['flesh'] = flesh
    df['FK'] = FK
    df.to_csv('readability_score.csv', index=False)
    return 0

readability_score()


