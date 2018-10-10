# -*- coding: utf-8 -*-
"""
Created on Sun Dec 03 16:33:31 2017

@author: ZQ
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import mat
import time

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

"""
np.linalg.matrix_rank(X_train)
"""

"""
classifier 0
"""
t0 = time.clock()

y_0 = np.zeros_like(y_train)
for i in range(y_train.shape[0]):
    if(y_train[i]==0):
        y_0[i] = 1
    else:
        y_0[i] = -1

lamda = 0.01
w_0 = np.dot(np.dot(mat((np.dot(X_train.T,X_train)+lamda*np.identity(784))).I,X_train.T),y_0).T
"""
y_0_test_calculated = np.dot(X_test,w_0)

y_0_test_classified = np.zeros_like(y_test)

for i in range(y_test.shape[0]):
    if(y_0_test_calculated[i]>0):
        y_0_test_classified[i] = 1
    else:
        y_0_test_classified[i] = -1

y_0_test = np.zeros_like(y_test)

for i in range(y_test.shape[0]):
    if(y_test[i]==0):
        y_0_test[i] = 1
    else:
        y_0_test[i] = -1
        
error = 0

for i in range(y_test.shape[0]):
    if(y_0_test_classified[i]!=y_0_test[i]):
        error+=1

error_rate = error/y_test.shape[0]
"""

"""
classifier 1
"""
y_1 = np.zeros_like(y_train)
for i in range(y_train.shape[0]):
    if(y_train[i]==1):
        y_1[i] = 1
    else:
        y_1[i] = -1

w_1 = np.dot(np.dot(mat((np.dot(X_train.T,X_train)+lamda*np.identity(784))).I,X_train.T),y_1).T

"""
classifier 2
"""
y_2 = np.zeros_like(y_train)
for i in range(y_train.shape[0]):
    if(y_train[i]==2):
        y_2[i] = 1
    else:
        y_2[i] = -1
        
w_2 = np.dot(np.dot(mat((np.dot(X_train.T,X_train)+lamda*np.identity(784))).I,X_train.T),y_2).T

"""
classifier 3
"""
y_3 = np.zeros_like(y_train)
for i in range(y_train.shape[0]):
    if(y_train[i]==3):
        y_3[i] = 1
    else:
        y_3[i] = -1

w_3 = np.dot(np.dot(mat((np.dot(X_train.T,X_train)+lamda*np.identity(784))).I,X_train.T),y_3).T

"""
classifier 4
"""
y_4 = np.zeros_like(y_train)
for i in range(y_train.shape[0]):
    if(y_train[i]==4):
        y_4[i] = 1
    else:
        y_4[i] = -1

w_4 = np.dot(np.dot(mat((np.dot(X_train.T,X_train)+lamda*np.identity(784))).I,X_train.T),y_4).T

"""
classifier 5
"""
y_5 = np.zeros_like(y_train)
for i in range(y_train.shape[0]):
    if(y_train[i]==5):
        y_5[i] = 1
    else:
        y_5[i] = -1

w_5 = np.dot(np.dot(mat((np.dot(X_train.T,X_train)+lamda*np.identity(784))).I,X_train.T),y_5).T

"""
classifier 6
"""
y_6 = np.zeros_like(y_train)
for i in range(y_train.shape[0]):
    if(y_train[i]==6):
        y_6[i] = 1
    else:
        y_6[i] = -1

w_6 = np.dot(np.dot(mat((np.dot(X_train.T,X_train)+lamda*np.identity(784))).I,X_train.T),y_6).T

"""
classifier 7
"""
y_7 = np.zeros_like(y_train)
for i in range(y_train.shape[0]):
    if(y_train[i]==7):
        y_7[i] = 1
    else:
        y_7[i] = -1

w_7 = np.dot(np.dot(mat((np.dot(X_train.T,X_train)+lamda*np.identity(784))).I,X_train.T),y_7).T

"""
classifier 8
"""
y_8 = np.zeros_like(y_train)
for i in range(y_train.shape[0]):
    if(y_train[i]==8):
        y_8[i] = 1
    else:
        y_8[i] = -1

w_8 = np.dot(np.dot(mat((np.dot(X_train.T,X_train)+lamda*np.identity(784))).I,X_train.T),y_8).T

"""
classifier 9
"""
y_9 = np.zeros_like(y_train)
for i in range(y_train.shape[0]):
    if(y_train[i]==9):
        y_9[i] = 1
    else:
        y_9[i] = -1

w_9 = np.dot(np.dot(mat((np.dot(X_train.T,X_train)+lamda*np.identity(784))).I,X_train.T),y_9).T

"""
test
"""
y_0_test_calculated = np.dot(X_test,w_0)
y_1_test_calculated = np.dot(X_test,w_1)
y_2_test_calculated = np.dot(X_test,w_2)
y_3_test_calculated = np.dot(X_test,w_3)
y_4_test_calculated = np.dot(X_test,w_4)
y_5_test_calculated = np.dot(X_test,w_5)
y_6_test_calculated = np.dot(X_test,w_6)
y_7_test_calculated = np.dot(X_test,w_7)
y_8_test_calculated = np.dot(X_test,w_8)
y_9_test_calculated = np.dot(X_test,w_9)
y_test_calculated = zip(y_0_test_calculated,y_1_test_calculated,y_2_test_calculated,y_3_test_calculated,y_4_test_calculated,y_5_test_calculated,y_6_test_calculated,y_7_test_calculated,y_8_test_calculated,y_9_test_calculated)
y_test_classified = np.argmax(y_test_calculated,axis = 1)

error = 0

for i in range(y_test.shape[0]):
    if(y_test_classified[i]!=y_test[i]):
        error+=1

error_rate = error/float(y_test.shape[0])
t1 = time.clock() - t0