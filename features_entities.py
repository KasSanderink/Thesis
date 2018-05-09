# Entity parsing with Spacy
###############################################################################

# Import standard libs
from collections import Counter
from time import time
import numpy as np

# Import self-written libs
from generate_documents import file_generator

# Import Spacy libs
import spacy

def get_tagset():
    tagset = {'PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 
               'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
               'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'FAC'}
    return tagset

def get_entities():
    gen = file_generator()
    tagset = get_tagset()
    n_features = len(tagset) + 2
    n_instances = 135 # sum(1 for i in gen) # 18510 # sum(1 for i in gen) # 185618 
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

get_entities()

