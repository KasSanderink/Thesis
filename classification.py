# Classification of a text-based dataset using a pipeline.
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
from sklearn.metrics import accuracy_score

# Import self-written scripts
from import_data import import_data

# Initialize data. Then split it in a train and a test set
t0 = time()
print("Loading data...")
raw_data = import_data()
X_train, X_test, y_train, y_test = train_test_split(raw_data['raw_text'], 
                                                    raw_data['target'])
print("Done, {0:.1f}s\n".format(time()-t0))

# Train the model (that is, the pipeline)
t0 = time()
print("Training...")
pipeline = Pipeline([('tfidf', TfidfVectorizer()), 
                     ('tree', DecisionTreeClassifier())])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print("Done, {0:.1f}s\n".format(time()-t0))

# Evaluate predictions
score = accuracy_score(y_test, predictions)
print("Accuracy: {}".format(score))