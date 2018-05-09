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

def scatter_plot(file, f1, f2):
    path = os.getcwd()
    data = pd.read_csv(file)
    data = data[pd.notnull(data)] # Remove NaN
    target = data['target']
    colors = ['blue', 'red', 'purple', 'green', 'orange']
    labels = ['aca', 'fic', 'mag', 'new', 'spo']
    fig, ax = plt.subplots()
    for i in range(5):
        X_genre = data[(data['target']) == i][f1]
        Y_genre = data[(data['target']) == i][f2]
        ax.scatter(X_genre, Y_genre, c=colors[i], label=labels[i])
    ax.legend()
    plt.xlabel('Relative noun occurence')
    plt.ylabel('Relative pronoun occurence')
    plt.show()

# noun_pronoun_data()
scatter_plot(os.getcwd() + '/preprocessed/COCA/POS_noun_pronoun.csv',
             'rel_pro', 'rel_noun')