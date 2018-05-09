# Classification of a text-based dataset using a pipeline. Pipeline consits of 
# a tf-idf vectorizer and a random forrest classifier.
###############################################################################

# Basic libraries
import numpy as np
from time import time

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Import self-written scripts
from import_documents import import_documents

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
                         ('tree', RandomForestClassifier())])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print("Done, {0:.1f}s".format(time()-t0))
    score = accuracy_score(y_test, predictions)
    print("Accuracy: {}".format(score))
    return 0

main()