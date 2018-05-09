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

get_POS_features()