# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 21:50:14 2017

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

seed = 782
np.random.seed(seed)

"""
import training data
"""
data_set = pd.read_csv("train.csv")
training_data = data_set.as_matrix()

label = training_data[:,0].astype('int8')
X = training_data[:,1:].astype('float64')

"""
print("Shape Train Images: (%d,%d)" % X.shape)
print("Shape Labels: (%d)" % label.shape) 
"""

"""
test data
"""

"""
Visualization of the data
"""
plt.figure(figsize=(12,12))
x = 3
y = 3
for i in range(0,y*x):
    plt.subplot(y,x,i+1)
    rdm = np.random.randint(0,X.shape[0],1)[0]
    img = np.reshape(X[rdm],(28,28))
    plt.imshow(img,cmap = "gray",interpolation = 'none')
    plt.title(str(label[rdm]))
plt.show()

"""
Count amount of samples per digit
"""
hist = np.ones(10)
for y in label:
    hist[y]+=1
colors = []
for i in range(10):
    level = 0.05*i;
    colors.append(plt.get_cmap("gray")(level))
bar = plt.bar(np.arange(10),hist,0.8,color = colors);
plt.grid()
plt.show()

"""
Normalization
"""
X = X/255
"""
print("Max of X: %.2f" % np.max(X))
print("Min of X: %.2f" % np.min(X))
"""

"""
One hot encoding
useful for classification
"""
label = pd.get_dummies(label).as_matrix()

"""
Activation function (Relu)
"""
def Relu(x, derivative=False):
    if(derivative == False):
        return x*(x>0)
    return 1*(x>0)
"""
x = np.arange(20)-10
relu = Relu(x)
plt.plot(x,relu)
plt.show()
"""

"""
Activation function (Softmax function)
"""
def Softmax(x):
    x -= np.max(x)
    return ((np.exp(x).T)/np.sum(np.exp(x),axis = 1)).T
"""
x = np.arange(20)-10
softmax = Softmax(x)
plt.plot(x,softmax)
plt.show()
"""

"""x
Create weights
create initial weights and bias with normal distribution
"""
def CreateWeights():
    number_inputs = 784
    number_hidenl1 = 500
    number_hidenl2 = 300
    number_outputs = 10
    
    w1 = np.random.normal(0,number_inputs**-0.5,[number_inputs,number_hidenl1])
    b1 = np.random.normal(0,number_inputs**-0.5,[1,number_hidenl1])
    w2 = np.random.normal(0,number_hidenl1**-0.5,[number_hidenl1,number_hidenl2])
    b2 = np.random.normal(0,number_hidenl1**-0.5,[1,number_hidenl2])
    w3 = np.random.normal(0,number_hidenl2**-0.5,[number_hidenl2,number_outputs])
    b3 = np.random.normal(0,number_hidenl2**-0.5,[1,number_outputs])
    
    return [w1,w2,w3,b1,b2,b3]

"""
Dropout
randomly drop neurons to prevent overfitting
"""
def Dropout(x,drop_prop):
    mask = np.random.binomial([np.ones_like(x)],1-drop_prop)[0]/(1-drop_prop)
    return x*mask

"""
Classify
"""
def Classify(weigths_bias,inputs,drop_prop=0):
    w1,w2,w3,b1,b2,b3 = weigths_bias
    """
    Matrix multiplication!!!!
    """
    firstHLoutput = Relu(np.dot(inputs,w1)+b1)
    firstHLoutput = Dropout(firstHLoutput,drop_prop)
    
    secondHLoutput = Relu(np.dot(firstHLoutput,w2)+b2)
    secondHLoutput = Dropout(secondHLoutput,drop_prop)
    
    finalOutput = Softmax(np.dot(secondHLoutput,w3)+b3)
    
    return [firstHLoutput,secondHLoutput,finalOutput]

"""
Accurate rate
"""
def AccurateRate(output,label):
    hit = 0
    outputMaxIndex = np.argmax(output,axis = 1)
    labelMaxIndex = np.argmax(label,axis = 1)
    for i in range(outputMaxIndex.shape[0]):
        if(outputMaxIndex[i]==labelMaxIndex[i]):
            hit+=1
    accurateRate = hit*100/output.shape[0]
    return accurateRate

"""
Cross entropy loss function
"""
def Loge_scalar(x):
    if(x!=0):
        return np.log(x)
    else:
        return -np.inf
    
def Loge_matrix(y):
    return [[Loge_scalar(xx) for xx in x] for x in y]

def Loss(classified,right):
    loss = -np.mean(np.nan_to_num(right*Loge_matrix(classified))+np.nan_to_num((1-right)*Loge_matrix(1-classified)),keepdims=True)
    return loss

"""
Cross validation
"""
test_percent = 0.1
test_size = int(round(X.shape[0]*test_percent))
index = np.arange(X.shape[0])
np.random.shuffle(index)
X_test = X[index[:test_size]]
y_test = label[index[:test_size]]
X_train = X[index[test_size:]]
y_train = label[index[test_size:]]


"""
Image transformation
Elastic transformation
"""
def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)
"""
X_et = np.array([elastic_transform(xx.reshape(28,28),15,3).reshape(784) for xx in X_train[0:10]])
plt.subplot(1,2,1)
plt.imshow(X_et[0].reshape(28,28))
plt.subplot(1,2,2)
plt.imshow(X_et[0].reshape(28,28))
"""
X_et = np.array([elastic_transform(xx.reshape(28,28),15,3).reshape(784) for xx in X_train[0:10]])
plt.subplot(4,1,1)
plt.title("5")
plt.imshow(X_et[0].reshape(28,28))
plt.subplot(4,1,2)
plt.title("0")
plt.imshow(X_et[1].reshape(28,28))
plt.subplot(4,1,3)
plt.title("3")
plt.imshow(X_et[2].reshape(28,28))
plt.subplot(4,1,4)
plt.title("2")
plt.imshow(X_et[3].reshape(28,28))
"""
SGD
"""
def SGD(weights,train_data,train_label,outputs,gamma,eta,lamda,momentum=None):
    w1,w2,w3,b1,b2,b3 = weights;
    if(momentum == None):
        vw1 = np.zeros_like(w1)
        vb1 = np.zeros_like(b1)
        vw2 = np.zeros_like(w2)
        vb2 = np.zeros_like(b2)
        vw3 = np.zeros_like(w3)
        vb3 = np.zeros_like(b3)
    else:
        vw1,vw2,vw3,vb1,vb2,vb3 = momentum
        
    firstHLoutput,secondHLoutput,finalOutput = outputs
    
    d1 = (train_label-finalOutput)/train_data.shape[0]
    delta_w3 = -np.dot(secondHLoutput.T,d1)+lamda*w3
    delta_b3 = -d1.sum(axis=0)+lamda*b3
    
    d2 = np.dot(d1,w3.T)*Relu(secondHLoutput,derivative=True)
    delta_w2 = -np.dot(firstHLoutput.T,d2)+lamda*w2
    delta_b2 = -d2.sum(axis=0)+lamda*b2
    
    d3 = np.dot(d2,w2.T)*Relu(firstHLoutput,derivative=True)
    delta_w1 = -np.dot(train_data.T,d3)+lamda*w1
    delta_b1 = -d3.sum(axis=0)+lamda*b1
    
    vw3 = gamma*vw3+eta*delta_w3
    vb3 = gamma*vb3+eta*delta_b3
    vw2 = gamma*vw2+eta*delta_w2
    vb2 = gamma*vb2+eta*delta_b2
    vw1 = gamma*vw1+eta*delta_w1
    vb1 = gamma*vb1+eta*delta_b1
    
    w3 = w3-vw3
    b3 = b3-vb3
    w2 = w2-vw2
    b2 = b2-vb2
    w1 = w1-vw1
    b1 = b1-vb1
    
    momentum = [vw1,vw2,vw3,vb1,vb2,vb3]
    weights = [w1,w2,w3,b1,b2,b3]
    
    return momentum,weights

"""
Training
"""
def train(weights,X_train,y_train,X_test,y_test,train_times = 10,train_size = 25,drop_prop = 0,gamma = 0,eta = 0.001,lamda = 0.001):
    momentum = None
    train_index = np.arange(X_train.shape[0])
    iteration_time = int(round(X_train.shape[0]/train_size))
    for i in range(train_times):
        np.random.shuffle(train_index)
        sum_loss = 0
        sum_acc = 0
        for j in range(iteration_time):
            sys.stdout.write("\nTrain times: %2d iteration times: %2d\n" % (i,j))
            start = j*train_size
            end = (j+1)*train_size
            if(end>X_train.shape[0]):
                end = X_train.shape[0]
            train_sample = np.array([elastic_transform(tmp.reshape(28,28),15,3).reshape(784) for tmp in X_train[train_index[start:end]]])
            train_label = y_train[train_index[start:end]]
            outputs = Classify(weights,train_sample,drop_prop)
            
            loss = Loss(outputs[-1],train_label)
            accuracyRate = AccurateRate(outputs[-1],train_label)
            sum_loss = sum_loss+loss
            ave_loss = sum_loss/(j+1)
            sum_acc = sum_acc+accuracyRate
            ave_acc = sum_acc/(j+1)
            sys.stdout.write("Train Size: %d loss: %f accuracy rate: %f" % (int(train_sample.shape[0]),ave_loss,ave_acc))
            
            momentum, weights= SGD(weights,train_sample,train_label,outputs,gamma,eta,lamda,momentum)
    
    outputs = Classify(weights,X_test,drop_prop)
    loss = Loss(outputs[-1],y_test)
    accuracyRate = AccurateRate(outputs[-1],y_test)
    
    return weights,loss,accuracyRate

weights = CreateWeights()
weigths,loss,accuracyRate = train(weights,X_train,y_train,X_test,y_test,train_times = 20,train_size = 100,drop_prop = 0.25,gamma = 0.9,eta = 0.1,lamda = 0.0000001)
sys.stdout.write("\rTest Size: %5d loss: %.4f accuracy rate: %.4f" % (X_test.shape[0],loss,accuracyRate))