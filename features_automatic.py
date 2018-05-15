# Generate new features. Contains POS-tagging, entity tagging and sentiment
###############################################################################

# Import basic libraries
from collections import Counter
import os
import re
import pandas as pd
import numpy as np
from time import time

# Import feature generating libs
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import ChartParser
import spacy

# Import self written scripts
from generate_documents import document_generator

# Generates a standard nltk tagset.
def get_tagset():
    path = os.getcwd()
    with open(path + '/data/tagset_nltk.txt', 'r') as file:
        data = file.read()
    tags_raw = re.findall("\n.{1,4}:", data)
    tags = [tag[1:-1] for tag in tags_raw]
    return tags

# Creates a 2D numpy array and writes it to a csv file. There's a try-except
# statement because sometimes the tagger uses a tag that's not in the tag-set
def get_POS_features():
    gen = document_generator()
    tagset = get_tagset()
    n_features = len(tagset) + 2
    n_instances = 18510 # sum(1 for i in gen) # 185618 
    result = np.empty([n_instances, n_features])
    for file in gen:
        empty_counter = {key: 0 for key in tagset}
        tags = nltk.pos_tag(nltk.word_tokenize(file[0]))
        tags_counter = Counter(tag for w,tag in tags)
        final_dict = {**empty_counter, **dict(tags_counter)}
        sorted_items = sorted(final_dict.items())
        keys = [item[0] for item in sorted_items]
        tag_count = [item[1] for item in sorted_items]
        tag_count.extend([file[2], file[1]])
        try:
            result[file[3]] = tag_count
        except:
            print(tag_count)
    np.savetxt("POS.csv", result, delimiter=",", fmt='%i')

# Requires two files: one with all POS-tag counts named 'POS.csv' and one with
# a word count named 'manual.csv'
def get_POS_relative():
    POS_path = os.getcwd() + "/preprocessed/COCA/POS.csv"
    manual_path = os.getcwd() + "/preprocessed/COCA/manual.csv"
    POS = pd.read_csv(POS_path)
    manual = pd.read_csv(manual_path)
    n_words = manual['n_words']
    target = POS['target']
    year = POS['year']
    tags = POS.drop(['target', 'year'], axis=1)
    for col in tags:
        POS[col] = POS[col] / n_words
        POS = POS.rename(columns = {col:col+'_rel'})
    POS.to_csv("POS_rel.csv", index=False)

# Possible entities
def get_entset():
    tagset = {'PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 
               'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
               'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'FAC'}
    return tagset

# Count entities in each document and produce a csv
def get_entities():
    gen = document_generator()
    tagset = get_entset()
    n_features = len(tagset) + 2
    n_instances = sum(1 for i in gen) # 18510 # sum(1 for i in gen) # 185618 
    gen = document_generator()
    result = np.empty([n_instances, n_features])
    nlp = spacy.load('en', disable=['tagger', 'parser', 'textcat'])
    for file in gen:
        empty_counter = {key: 0 for key in tagset}
        doc = nlp(file[0])
        entity_counter = Counter(ent.label_ for ent in doc.ents)
        final_dict = {**empty_counter, **dict(entity_counter)}
        sorted_items = sorted(final_dict.items())
        ent_count = [item[1] for item in sorted_items]
        ent_count.extend([file[2], file[1]])
        try:
            result[file[3]] = ent_count
        except:
            print(final_dict)
    result = np.nan_to_num(result)
    np.savetxt("entity.csv", result, delimiter=",", fmt='%i')

# Run get_entities() first.
def get_entities_relative():
    ent_path = os.getcwd() + "/preprocessed/COCA/entities.csv"
    manual_path = os.getcwd() + "/preprocessed/COCA/manual.csv"
    ent = pd.read_csv(ent_path)
    manual = pd.read_csv(manual_path)
    n_words = manual['n_words']
    target = ent['target']
    year = ent['year']
    tags = ent.drop(['target', 'year'], axis=1)
    for col in tags:
        ent[col] = ent[col] / n_words
        ent = ent.rename(columns = {col:col+'_rel'})
    ent.to_csv("ent_rel.csv", index=False)

def get_sentiment():
    gen = document_generator()
    SID = SentimentIntensityAnalyzer()
    n_features = 4 + 2
    n_instances = sum(1 for i in gen)
    print("Done counting...")
    gen = document_generator()
    result = np.empty([n_instances, n_features])
    for file in gen:
        t0 = time()
        s = SID.polarity_scores(file[0])
        scores = [s['neg'], s['neu'], s['pos'], s['compound']]
        result[file[3]] = scores + [file[2], file[1]]
    np.savetxt("sentiment.csv", result, delimiter=",", fmt='%f')

def get_tree_depth():
    gen = document_generator()
    n_features = 1
    n_instances = 17821# sum(1 for i in gen)
    gen = document_generator()
    result = np.empty([n_instances, n_features])
    nlp = spacy.load('en', disable=['tagger', 'entity', 'textcat'])
    for file in gen:
        t0 = time()
        sentences = nltk.sent_tokenize(file[0])
        for sent in sentences:
            doc = nlp(sent)
            for token in doc:
                if token.dep_ == 'ROOT':
                    print(max([]))
                #len(set(chunk.root.head.text))
        print(time()-t0)

get_tree_depth()
