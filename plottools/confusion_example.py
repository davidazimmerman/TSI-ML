import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from classificationplots import plot_confusion_matrix

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier
classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)

# Predict classes using test set
y_pred = classifier.predict(X_test)

# Map the integer codes for classes to actual class names
map_dict = {0:class_names[0], 1:class_names[1], 2:class_names[2]}
y_test = np.array(list(map(map_dict.get, y_test)))
y_pred = np.array(list(map(map_dict.get, y_pred)))

plot_confusion_matrix(y_test, y_pred, labels = class_names, savefig=True)