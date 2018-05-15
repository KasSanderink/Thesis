# Import libraries
import glob
import os
import pandas as pd
import re
import numpy as np
from stop_words import get_stop_words

# Generator for the COCA files
def document_generator():
    path = os.getcwd()
    folders = glob.glob(path+'/data/COCA/*')
    n_folders = len(folders)
    n_instance = -1
    for i in range(n_folders):
        folder = folders[i]
        target = i
        files = glob.glob(folder+'/*')
        n_files = len(files)
        for j in range(n_files):
            print(j)
            file = files[j]
            year = files[j].rsplit("_",1)[1][:4]
            with open (file, "r") as current_file:
                raw_text = current_file.read()
                if re.search('##\d\d\d\d\d\d\d', raw_text):
                    text = re.split('##\d\d\d\d\d\d\d', raw_text)
                else:
                    text = re.split('@@\d\d\d\d\d\d\d', raw_text)
                text = [item.replace(" @ @ @ @ @ @ @ @ @ @ ", " ") 
                        for item in text]
                n_strings = int(len(text)/2) # Shrink dataset
                for k in range(n_strings):
                    n_instance += 1
                    yield text[k], target, year, n_instance

# Turn COCA into structured data
def COCA_structured():
    gen = document_generator()
    n_instances = sum(1 for i in gen)
    print(n_instances)
    gen = document_generator()
    n_columns = 3
    result = np.empty([n_instances, n_columns], dtype=np.object)
    for file in gen:
        result[file[3]] = np.array([file[0], file[2], file[1]])
    result = pd.DataFrame(result, columns=['raw_text', 'year', 'target'])
    result = result[result['raw_text'] != ""]
    result.to_csv("COCAa_structured.csv", index=False)
COCA_structured()