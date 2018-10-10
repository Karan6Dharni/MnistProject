# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 21:15:50 2017

@author: ZQ
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import sys
import warnings
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier

seed = 782
np.random.seed(seed)

"""
import training data
"""
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
t0 = time.clock()
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=10)
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
t1 = time.clock() - t0