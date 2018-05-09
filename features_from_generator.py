# Functions that take a generator as input, and give a column of features as
# output
###############################################################################

import numpy as np
import pandas as pd
import os

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
def get_POS_features(gen, n_instances):
    tagset = get_tagset()
    n_features = len(tagset) + 2
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
    return result

# Get the average word length of a document. Words is a DataFrame. 
def get_word_length(gen, instances, words=None):
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

# It's essential that the number of colons has been counted
def get_spoken_flag(instances, colons):
    mean = np.mean(colons)
    result = np.empty([instances])
    for i in range(instances):
        if colons[i] > (mean * 5):
            result[i] = 10
        else:
            result[i] = 0
    return result

data = pd.read_csv(os.getcwd() + "/preprocessed/COCA/POS_all.csv")
colon = get_spoken_flag(18510, data[':'])
np.savetxt("prr.csv", colon, delimiter=',', fmt='%i')


