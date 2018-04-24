# Visualise interpunction
###############################################################################

# Import basic libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os

def barplot(mean_data, period, x_ticks):
    bar_width = 0.15
    i = 0
    x_labels = np.arange(len(mean_data.index))
    figure, ax = plt.subplots()
    for index, row in mean_data.iterrows():
        bar = ax.bar(x_labels + (i * bar_width) - 0.3, row, bar_width, 
                     label=str(index))
        i+=1
    plt.xticks(x_labels, x_ticks)
    plt.xlabel("Genre")
    plt.ylabel("Average of (n_chars/n_occurences) * 1000")
    plt.legend()
    plt.title("Relative occurence of several punctuation marks\n" + 
              "in different genres in " + period)
    plt.savefig("visuals\interpunction" + period + ".png")
    return 0

def get_means(data, features):
    targets = set(data['target'])
    result = pd.DataFrame()
    for target in targets:
        target_rows = data[data['target'] == target]
        result[target] = target_rows[features].apply(np.mean)
    return result

def load_data(path, header):
    data = pd.read_csv(path)
    data.columns = header
    return data

def generate_barplots():
    path = "\preprocessed\COCA\interpunction_relative"
    periods = ["1990-1995", "1995-2000", "2000-2005", "2005-2010", "2010-2015"]
    features = ['full_stop', 'colon', 'semi_colon', 'exclamation', 'question']
    header = ['full_stop', 'colon', 'semi_colon', 'exclamation',
                    'question', 'n_chars', 'year', 'target']
    x_ticks = ['academic', 'fiction', 'magazine', 'newspaper', 'spoken']
    for period in periods:
        data = load_data(os.getcwd()+path+period+".csv", header)
        mean_data = get_means(data, features)
        barplot(mean_data, period, x_ticks)
    return 0

generate_barplots()