# Classification of a text-based dataset using a pipeline. Pipeline consits of 
# a tf-idf vectorizer and a decision tree model. The actual decision tree is 
# exported to tree.pdf.
###############################################################################

# Currently unused imports
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer

# Basic libraries
import numpy as np
from time import time

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score

# Visualisation tools
from graphviz import Source

# Import self-written scripts
from import_documents import import_documents

def visualize_tree(tree_string):
    source = Source(tree_string)
    source.render('tree', view=True)
    return 0

# Initialize data. Then split it in a train and a test set, and train the model
def main():
    t0 = time()
    print("Loading data...")
    raw_data = import_documents()
    X_train, X_test, y_train, y_test = train_test_split(raw_data['raw_text'], 
                                                        raw_data['target'])
    print("Done, {0:.1f}s".format(time()-t0))
    t0 = time()
    print("Training...")
    pipeline = Pipeline([('tfidf', TfidfVectorizer(max_df=0.7)), 
                         ('tree', DecisionTreeClassifier())])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print("Done, {0:.1f}s".format(time()-t0))
    score = accuracy_score(y_test, predictions)
    print("Accuracy: {}".format(score))
    tree_string = export_graphviz(pipeline.named_steps['tree'], out_file=None)
    visualize_tree(tree_string)
    return 0

main()