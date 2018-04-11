# Assuming there's a folder called 'COCA' in the current working directiory 
# which contains some number of subfolders. The names of these folders are the 
# targets. The files in these folders are the instances of the dataset.
###############################################################################

# Import libraries
import glob
import os
import pandas as pd

# Turn a documunt (input is a path-name) into a string
def _read_file(file):
    with open (file, "r") as current_file:
        raw_text = current_file.read()
    return(raw_text)

# Each folder contains some number of documents. Turn documents into strings, 
# put strings in a list. Years are found by looking for the last "_" in the 
# path-name and taking the 4 characters that follow.
def _extract_data(folder):
    files = glob.glob(folder+'\*')
    n_files = len(files)
    text_strings = []
    years = []
    for i in range(5):
        raw_text = _read_file(files[i])
        text_strings.append(raw_text)
        years.append(files[i].rsplit("_",1)[1][:4])
    return text_strings, years, 5

# Return a DataFrame with the data. 
def import_data():
    path = os.getcwd()
    folders = glob.glob(path+'\COCA\*')
    n_folders = len(folders)
    data = {'raw_text':[],'year':[],'target':[]}
    for i in range(n_folders):
        text_strings, years, n_files = _extract_data(folders[i])
        data['raw_text'].extend(text_strings)
        data['year'].extend(years)
        data['target'].extend([i]*n_files)
    data = pd.DataFrame.from_dict(data)
    return data