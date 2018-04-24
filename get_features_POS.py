# Generate a new csv-file which contains the POS-tag count for each instance
###############################################################################

# Import basic libraries
from collections import Counter
import os
import re
import pandas as pd
import numpy as np
from time import time

# Import POS-tagging libraries
import nltk
import spacy as sp
from spacy.attrs import TAG

# Import self written scripts
from generate_documents import document_generator

# Get tags of one document. Returns a counter.
def get_tags(document):
    tags = nltk.pos_tag(nltk.word_tokenize(document))
    counter = Counter(tag for w,tag in tags)
    return counter

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
def get_POS_features_nltk():
    gen = document_generator()
    tagset = get_tagset()
    n_features = len(tagset) + 2
    n_instances = 185618 # sum(1 for i in gen)
    result = np.empty([n_instances, n_features])
    for file in gen:
        empty_counter = {key: 0 for key in tagset}
        t0 = time()
        tags = nltk.pos_tag(nltk.word_tokenize(file[0]))
        print("TAGGING:", time()-t0)
        tags_counter = Counter(tag for w,tag in tags)
        final_dict = {**empty_counter, **dict(tags_counter)}
        sorted_items = sorted(final_dict.items())
        tag_count = [item[1] for item in sorted_items]
        tag_count.extend([file[2], file[1]])
        try:
            result[file[3]] = tag_count
        except:
            print(tag_count)
        print("DONE:", time()-t0)
    np.savetxt("POS.csv", result, delimiter=",", fmt='%i')

# Almost the same as the nltk version. Pro: the tagset is definite. Con: it's
# super slow (for this reason I haven't finished the last part, so this 
# function doesn't really work)
def get_POS_features_spacy():
    nlp = sp.load('en')
    gen = document_generator()
    tagset = get_tagset()
    n_features = len(tagset) + 2
    n_instances = 185618 # sum(1 for i in gen)
    result = np.empty([n_instances, n_features])
    for file in gen:
        empty_counter = {key: 0 for key in tagset}
        t0 = time()
        doc = nlp(file[0])
        #tags_counter = doc.count_by(TAG)
        tags_counter = Counter(token.tag_ for token in doc)

get_POS_features_spacy()