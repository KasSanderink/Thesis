# Create a generator. Generates one instance at a time.
###############################################################################

# Import libraries
import glob
import os
import pandas as pd
import re
from stop_words import get_stop_words

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
                n_strings = int(len(text)/10) # Shrink dataset
                for k in range(n_strings):
                    n_instance += 1
                    yield text[k], target, year, n_instance

def file_generator():
    path = os.getcwd()
    folders = glob.glob(path+'/data/COCA/*')
    n_folders = len(folders)
    n_file = -1
    for i in range(n_folders):
        folder = folders[i]
        target = i
        files = glob.glob(folder+'/*')
        n_files = len(files)
        for j in range(n_files):
            file = files[j]
            year = files[j].rsplit("_",1)[1][:4]
            with open (file, "r") as current_file:
                raw_text = current_file.read()
                n_file += 1
                yield raw_text[:100000], target, year, n_file