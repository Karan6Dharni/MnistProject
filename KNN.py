# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 22:06:33 2017

@author: ZQ
"""

from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
sns.set_style("white")

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure


data_set = pd.read_csv("train.csv")
training_data = data_set.as_matrix()

label = training_data[:,0].astype('int8')
X = training_data[:,1:].astype('float64')

test_percent = 0.1
test_size = int(round(X.shape[0]*test_percent))
index = np.arange(X.shape[0])
np.random.shuffle(index)
X_test = X[index[:test_size]]
y_test = label[index[:test_size]]
X_train = X[index[test_size:]]
y_train = label[index[test_size:]]

kVals = range(1, 30, 2)
accuracies = []
 
# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in xrange(1, 30, 2):
	# train the k-Nearest Neighbor classifier with the current value of `k`
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(X_train, y_train)
 
	# evaluate the model and update the accuracies list
	score = model.score(X_test, y_test)
	print("k=%d, accuracy=%.2f%%" % (k, score * 100))
	accuracies.append(score)
 
# find the value of k that has the largest accuracy
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
	accuracies[i] * 100))
