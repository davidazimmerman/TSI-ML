import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from classificationplots import plot_roc_curve

# Load in data
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict classes using test set
y_pred = classifier.predict_proba(X_test)

plot_roc_curve(y_test, y_pred[:,1], savefig=True)