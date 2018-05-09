# Use regex to determine if there's APA references in an text. If there are,
# it's probably from an academic journal.
###############################################################################

# Import self-written libs
from generate_documents import document_generator

# Import basic libs
import numpy as np
import re
from collections import Counter

def count_APA():
    gen = document_generator()
    n_instances = 185618 # sum(1 for i in gen)
    n_features = 3 + 2
    result = np.empty([n_instances, n_features])
    for file in gen:
        apa_1 = len(re.findall("\( [A-Z][a-z]* , \d\d\d\d \)", file[0]))
        et_al = len(re.findall("et\.* al",  file[0]))                     
        year = len(re.findall(" \d\d\d\d ", file[0]))
        instance = [apa_1, apa_2, year, file[2], file[1]]
        result[file[3]] = instance
    np.savetxt("apa.csv", result, delimiter=",", fmt='%i')



        

count_APA()