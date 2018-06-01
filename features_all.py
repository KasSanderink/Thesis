import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from time import time

import pyphen
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
import spacy

from generate_documents import document_generator

# Define variables
min_len = 100

def main():

    # Initialize objects
    gen = document_generator()
    nlp = spacy.load('en', disable=['tagger', 'parser', 'textcat'])
    dic = pyphen.Pyphen(lang='en')
    tok = RegexpTokenizer('\w+')
    sid = SentimentIntensityAnalyzer()
    genre_result = []

    # Tagsets
    POS_tags = {"''", '(', ')', ',', '--', '.', ':', 'CC', 'CD', 'DT', 'EX', 
                 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 
                 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 
                 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 
                 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``', '$', '#'}
    entity_tags = {'PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 
                  'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
                  'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'FAC'}

    # Generate features
    # TODO When the generator yields 0 save the progress
    for file in gen:

        # Save file if all the documents from a genre are generated
        if 'genre' == file[0]:
            genre_result = np.array(genre_result)
            genre_result = pd.DataFrame(data=genre_result)
            genre_result.to_csv('/datastore/10814418/preprocessed_' + 
                                str(file[1]) + '.csv', index=False)
            genre_result = []
            continue

        # Check file if the file is non-empty and name variables
        text = file[0]
        info = np.array([file[2], file[1]])
        index = file[3]
        if len(text) < min_len:
            continue

        # POS-tags
        empty_counter = {key: 0 for key in POS_tags}
        tags = nltk.pos_tag(nltk.word_tokenize(text))
        tags_counter = Counter(tag for w,tag in tags)
        final_dict = {**empty_counter, **dict(tags_counter)}
        sorted_items = sorted(final_dict.items())
        keys = [item[0] for item in sorted_items]
        tag_count = np.array([item[1] for item in sorted_items])
        
        # Entities
        empty_counter = {key: 0 for key in entity_tags}
        doc = nlp(text)
        entity_counter = Counter(ent.label_ for ent in doc.ents)
        final_dict = {**empty_counter, **dict(entity_counter)}
        sorted_items = sorted(final_dict.items())
        ent_count = np.array([item[1] for item in sorted_items])

        # Sentence, word and syllable count
        n_sent = len(nltk.sent_tokenize(text)) 
        words = tok.tokenize(text)
        n_word = len(words)
        syllables = [dic.inserted(word) for word in words]
        syllable_list = [len(re.findall('-', word)) + 1 for word in syllables] 
        n_syl = sum(syllable_list)
        syntax_count = np.array([n_sent, n_word, n_syl])

        # Readability score
        flesh = 206.835-1.015*(n_word/n_sent)-84.6*(n_syl/n_word)
        flesh_kincaid = 0.39*(n_word/n_sent)+11.8*(n_syl/n_word)-15.59
        readability_score = np.array([flesh, flesh_kincaid])

        # Sentiment
        score_dic = sid.polarity_scores(text)
        sentiment = np.array([score_dic['neg'], score_dic['neu'], 
                              score_dic['pos'], score_dic['compound']])

        # Concat all features
        instance_result = np.concatenate([tag_count, ent_count, syntax_count, 
                                          readability_score, sentiment, info])
        genre_result.append(instance_result)

main()
