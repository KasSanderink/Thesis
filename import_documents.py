# Assuming there's a folder called 'COCA' in the current working directiory 
# which contains some number of subfolders. The names of these folders are the 
# targets. The files in these folders are the instances of the dataset.
###############################################################################

# Import libraries
import glob
import os
import pandas as pd
import re
from stop_words import get_stop_words

stop_words = get_stop_words('en')
max_strings = 5000


# Stem and remove stop words
def _strip(raw_text):
    return raw_text

# Turn a documunt (input is a path-name) into a string
def _read_file(file, strip):
    with open (file, "r") as current_file:
        raw_text = current_file.read()
        if re.search('##\d\d\d\d\d\d\d', raw_text):
            text = re.split('##\d\d\d\d\d\d\d', raw_text)[:max_strings]
        else:
            text = re.split('@@\d\d\d\d\d\d\d', raw_text)[:max_strings]
        text = [item.replace(" @ @ @ @ @ @ @ @ @ @ ", " ") for item in text]
        return (_strip(text), len(text)) if strip else (text, len(text))

# Each folder contains some number of documents. Turn documents into strings, 
# put strings in a list. Years are found by looking for the last "_" in the 
# path-name and taking the 4 characters that follow.
def _extract_data(folder, strip):
    files = glob.glob(folder+'\*')
    n_files = 5 # len(files)
    text_strings = []
    years = []
    n_strings = 0
    for i in range(n_files):
        raw_text, n_file_strings = _read_file(files[i], strip)
        text_strings.extend(raw_text)
        years.extend([files[i].rsplit("_",1)[1][:4]] * n_file_strings)
        n_strings += n_file_strings
    return text_strings, years, n_strings

# Return a DataFrame with the data. Again, it assumes there's a folder named 
# COCA which contains the dataset in a specific format. 
def import_documents(strip=True):
    path = os.getcwd()
    folders = glob.glob(path+'\COCA\*')
    n_folders = len(folders)
    data = {'raw_text':[],'year':[],'target':[]}
    for i in range(n_folders):
        text_strings, years, n_strings = _extract_data(folders[i], strip)
        data['raw_text'].extend(text_strings)
        data['year'].extend(years)
        data['target'].extend([i]*n_strings)
    data = pd.DataFrame.from_dict(data)
    return data