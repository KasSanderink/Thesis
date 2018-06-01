import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

import pyphen
import lightgbm as lgb
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale
from sklearn.externals import joblib
import spacy

from generate_documents import document_generator
from visualise_scatter import scatter_plot

def make_relative(X):
    for i in range(65):
        X[:,i] = X[:,i] / X[:,66]
    return X

def init_lgb():
    clf = LGBMClassifier(objective='multiclass',
                         num_class=4,
                         num_leaves=127,
                         min_data_in_leaf=200,
                         n_estimators=130)
    return clf

def get_feature_names():
    names = {'#','$',"''",'(',')','comma','--','.',':','CC','CD','DT','EX',
             'FW','IN','JJ','JJR','JJS','LS','MD','NN','NNP','NNPS','NNS',
             'PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH',
             'VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','``',
             'CARDINAL','DATE','EVENT','FAC','FACILITY','GPE','LANGUAGE','LAW',
             'LOC','MONEY','NORP','ORDINAL','ORG','PERCENT','PERSON','PRODUCT',
             'QUANTITY','TIME','WORK_OF_ART','n_sent','n_word','n_syl',
             'flesh','flesh_kincaid','neg','neu','pos','compound'}
    return names

def generate_features(text):

    # Initialize objects
    gen = document_generator()
    nlp = spacy.load('en', disable=['tagger', 'parser', 'textcat'])
    dic = pyphen.Pyphen(lang='en')
    tok = RegexpTokenizer('\w+')
    sid = SentimentIntensityAnalyzer()
    genre_result = []

    # Tagsets
    POS_tags = {"''", '(', ')', ',', '--', '.', ':', 'CC', 'CD', 'DT', 'EX', 
                 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 
                 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 
                 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 
                 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``', '$', '#'}
    entity_tags = {'PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 
                  'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
                  'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'FAC'}

    # POS-tags
    empty_counter = {key: 0 for key in POS_tags}
    tags = nltk.pos_tag(nltk.word_tokenize(text))
    tags_counter = Counter(tag for w,tag in tags)
    final_dict = {**empty_counter, **dict(tags_counter)}
    sorted_items = sorted(final_dict.items())
    keys = [item[0] for item in sorted_items]
    tag_count = np.array([item[1] for item in sorted_items])
    
    # Entities
    empty_counter = {key: 0 for key in entity_tags}
    doc = nlp(text)
    entity_counter = Counter(ent.label_ for ent in doc.ents)
    final_dict = {**empty_counter, **dict(entity_counter)}
    sorted_items = sorted(final_dict.items())
    ent_count = np.array([item[1] for item in sorted_items])

    # Sentence, word and syllable count
    n_sent = len(nltk.sent_tokenize(text)) 
    words = tok.tokenize(text)
    n_word = len(words)
    syllables = [dic.inserted(word) for word in words]
    syllable_list = [len(re.findall('-', word)) + 1 for word in syllables] 
    n_syl = sum(syllable_list)
    syntax_count = np.array([n_sent, n_word, n_syl])

    # Readability score
    flesh = 206.835-1.015*(n_word/n_sent)-84.6*(n_syl/n_word)
    flesh_kincaid = 0.39*(n_word/n_sent)+11.8*(n_syl/n_word)-15.59
    readability_score = np.array([flesh, flesh_kincaid])

    # Sentiment
    score_dic = sid.polarity_scores(text)
    sentiment = np.array([score_dic['neg'], score_dic['neu'], 
                          score_dic['pos'], score_dic['compound']])

    # Concat all features
    instance_result = np.concatenate([tag_count, ent_count, syntax_count, 
                                      readability_score, sentiment])
    return instance_result

# Load a sample text file
def load_file(filename):
    genres = {'academic':0,'fiction':1,'magazine':2,'newspaper':3}
    path = os.getcwd() + '/data/samples/' + filename
    with open(path, 'r') as file:
        text = file.read()
    target, year, _ = filename.split('_')
    return text, year, target

# Show a genre against all other genres. Plots with the 10 most descriptive
# features
def visualise_binary(prediction, features, data):
    genres = ['academic', 'fiction', 'magazine', 'newspaper']
    genre = genres[prediction]
    all_features = pd.read_csv(os.getcwd() + 
                                     '/results/binary_feature_importance.csv')
    genre_features = all_features[genre]
    for i in range(0, 10, 2):
        pair = genre_features[i:i + 2]
        scatter_plot(data, pair.iloc[0], pair.iloc[1], prediction, features)
    return 0

# For each target, get all the instances. Then get the average of each feature.
# Now you've got a series with features as indexes. This series is appended
# as an array to an array of features. The shape is (n_features, 2). 
# Now normalize each pair of values. This is not working very well.
def visualise(prediction, features, feature_names, feature_dict, data):
    targets = list(range(4))
    targets.remove(prediction)
    target_instances = data[data['target'] == prediction]
    for i in targets:
        genre_i = data[data['target'] == i]
        feature_avg = genre_i.apply(np.mean).drop(['target', 'year']).values
        norm_avg = minmax_scale(np.column_stack((feature_avg, features)))
        dist = abs(norm_avg[:,0] - norm_avg[:,1])
        f1, f2 = [feature_names[i] for i  in dist.argsort()[-2:][::-1]]
        fig, ax = plt.subplots()
        ax.scatter(genre_i[f1], genre_i[f2], c='blue')
        ax.scatter(target_instances[f1], target_instances[f2], c='orange')
        plt.show()
    return 0

# Very hard-coded (that is the point, in some sense)
def factualise(prediction, feature_dict, data):
    avg = data.apply(np.mean)
    if prediction == 3:
        if feature_dict['DATE_rel'] > avg['DATE_rel']:
            num = feature_dict['DATE_rel'] / avg['DATE_rel']
            print("Dates occured {:.2f} times more than usual.".format(num))

def explain(result, features, feature_names, feature_dict):
    prediction = np.argmax(result[0])
    data = pd.read_csv(os.getcwd() + '/preprocessed/COCA/all_features.csv')
    data = data.dropna()
    factualise(prediction, feature_dict, data)
    visualise(prediction, features, feature_names, feature_dict, data)
    return 0

# TODO remove features that LGB does not use from feature dict
def main():
    #model = lgb.Booster(model_file='lgb_model.txt')
    model = joblib.load('filename.pkl')
    text, year, target = load_file('academic_2013_1.txt')
    features = generate_features(text).reshape((1,-1))
    result = model.predict(features)
    prediction = np.argmax(result[0])
    #feature_names = model.feature_name()
    #features = features[0]
    #n_features = len(features)
    #feature_dict = {feature_names[i]:features[i] for i in range(n_features)}
    #print(feature_dict)
    print(result)
    # explain(result, features, feature_names, feature_dict)
    return 0

main()