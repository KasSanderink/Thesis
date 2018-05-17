import os
import re
import numpy as np
import pandas as pd
from collections import Counter

import pyphen
import lightgbm as lgb
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
import spacy

from generate_documents import document_generator
from visualise_scatter import scatter_plot

def get_tagset():
    path = os.getcwd()
    with open(path + '/data/tagset_nltk.txt', 'r') as file:
        data = file.read()
    tags_raw = re.findall("\n.{1,4}:", data)
    tags = [tag[1:-1] for tag in tags_raw]
    return tags

def get_POS_relative(text):
    tagset = get_tagset()
    empty_counter = {key: 0 for key in tagset}
    tags = nltk.pos_tag(nltk.word_tokenize(text))
    tags_counter = Counter(tag for w,tag in tags)
    final_dict = {**empty_counter, **dict(tags_counter)}
    sorted_items = sorted(final_dict.items())
    keys = [item[0] for item in sorted_items]
    tag_count = np.array([item[1] for item in sorted_items])
    return tag_count / sum(tag_count)

# Possible entities
def get_entset():
    tagset = {'PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 
               'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
               'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'FAC'}
    return tagset

def get_entity_relative(text):
    tagset = get_entset()
    nlp = spacy.load('en', disable=['tagger', 'parser', 'textcat'])
    empty_counter = {key: 0 for key in tagset}
    doc = nlp(text)
    entity_counter = Counter(ent.label_ for ent in doc.ents)
    final_dict = {**empty_counter, **dict(entity_counter)}
    sorted_items = sorted(final_dict.items())
    ent_count = np.array([item[1] for item in sorted_items])
    return ent_count / sum(ent_count)

def get_sentiment(text):
    SID = SentimentIntensityAnalyzer()
    dic = SID.polarity_scores(text)
    scores = [dic['neg'], dic['neu'], dic['pos'], dic['compound']]
    return np.array(scores)

def get_readability(text):
    dic = pyphen.Pyphen(lang='en')
    tok = RegexpTokenizer('\w+')
    sentence_count = len(nltk.sent_tokenize(text)) 
    words = tok.tokenize(text)
    word_count = len(words)
    syllables = [dic.inserted(word) for word in words]
    syllable_count = [len(re.findall('-', word)) + 1 for word in syllables] 
    syllable_count = sum(syllable_count)
    arr = [sentence_count, word_count, syllable_count]
    flesh = 206.835-1.015*(arr[1]/arr[0])-84.6*(arr[2]/arr[1])
    FK = 0.39*(arr[1]/arr[0])+11.8*(arr[2]/arr[1])-15.59
    return np.array([FK, flesh])


def generate_features(text):
    POS_relative = get_POS_relative(text)
    entity_relative = get_entity_relative(text)
    sentiment = get_sentiment(text)
    readability = get_readability(text)
    features = np.concatenate([readability, sentiment, entity_relative, 
                               POS_relative])
    return features

def load_file(filename):
    genres = {'academic':0,'fiction':1,'magazine':2,'newspaper':3}
    path = os.getcwd() + '/data/samples/' + filename
    with open(path, 'r') as file:
        text = file.read()
    return text, filename[-8:-4], genres[filename[:-9]]

def visualise(prediction, features, data):
    genres = ['academic', 'fiction', 'magazine', 'newspaper']
    genre = genres[prediction]
    all_features = pd.read_csv(os.getcwd() + 
                                     '/results/binary_feature_importance.csv')
    genre_features = all_features[genre]
    for i in range(0, 10, 2):
        pair = genre_features[i:i + 2]
        scatter_plot(data, pair.iloc[0], pair.iloc[1], prediction, features)
    return 0

# Very hard-coded (that is the point, in some sense)
def factualise(prediction, features, data):
    avg = data.apply(np.mean)
    if prediction == 3:
        if features['DATE_rel'] > avg['DATE_rel']:
            num = features['DATE_rel'] / avg['DATE_rel']
            print("Dates occured {:.2f} times more than usual.".format(num))


def explain(result, features):
    prediction = np.argmax(result[0])
    data = pd.read_csv(os.getcwd() + '/preprocessed/COCA/all_features.csv')
    factualise(prediction, features, data)
    visualise(prediction, features, data)
    return 0

def main():
    model = lgb.Booster(model_file='lgb_model.txt')
    text, year, target = load_file('academic_2013.txt')
    features = generate_features(text).reshape((1,-1))
    result = model.predict(features)
    feature_names = model.feature_name()
    features = features[0]
    n_features = len(features)
    features = {feature_names[i]:features[i] for i in range(n_features)}
    explain(result, features)
    return 0

main()