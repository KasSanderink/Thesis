import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def noun_pronoun_data():
    path = os.getcwd()
    data = pd.read_csv(path + '/preprocessed/COCA/POS.csv')
    target = data['target']
    year= data['year']
    data = data.drop(['target', 'year'], 1)
    nouns = data[['NN', 'NNP','NNPS', 'NNS', ]]
    pronouns = data[['PRP', 'PRP$']]
    n_words = data.apply(np.sum, axis=1)
    n_nouns = nouns.apply(np.sum, axis=1)
    n_pronouns = pronouns.apply(np.sum, axis=1)
    relative_nouns = n_nouns/n_words
    relative_pronouns = n_pronouns/n_words
    result = pd.concat([n_words, n_nouns, n_pronouns, relative_nouns, 
                        relative_pronouns, year, target], axis=1)
    result.columns = ['n_words', 'n_nouns', 'n_pronouns', 'rel_noun','rel_pro',
                      'year','target']

    result.to_csv(path+'/preprocessed/COCA/POS_noun_pronoun.csv', index=False)

# file is a csv with features, f1 and f2 are the features to be plotted. Binary
# can be None, or a number. If it's a number the corresponding genre will 
# occur in red, and all the other genres in blue (making a X vs not-X 
# distinction). The binary part is a bit awkward but it works.
def scatter_plot(file, f1, f2, binary=None):
    path = os.getcwd()
    data = pd.read_csv(file)
    data = data[pd.notnull(data)] # Remove NaN
    target = data['target']
    colors = ['blue', 'green', 'orange', 'red']
    genres = [0, 1, 2, 3]
    if binary != None:
        colors = ['blue'] * 4
        colors[binary] = 'red'
        genres.remove(binary)
        genres.append(binary)
    labels = ['academic', 'fiction', 'magazine', 'newspaper']
    fig, ax = plt.subplots()
    for i in genres:
        X_genre = data[(data['target']) == i][f1]
        Y_genre = data[(data['target']) == i][f2]
        ax.scatter(X_genre, Y_genre, c=colors[i], label=labels[i])
    ax.legend()
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.show()

# noun_pronoun_data()
scatter_plot(os.getcwd() + '/preprocessed/COCA/POS_relative.csv',
             '._rel', ':_rel', binary=0)