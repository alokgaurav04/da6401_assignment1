import numpy as np
#Activation functions

def sigmoid(x):
    #Clip for Math overflow
    x = np.clip(x, a_min = -10**3, a_max = 10*3)
    return (1/(1+ np.exp(-x)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    exp=np.exp(x-np.max(x,axis=1,keepdims=True))
    return exp/np.sum(exp,axis=1,keepdims=True)

#Calculate derivative of Activation functions to be used in the Backpropogation stage.
def relu_derivative(x):
    return (x>0).astype(float)

def sigmoid_derivative(a):
    h = sigmoid(a)
    return h*(1-h)
  
def tanh_derivative(a):
    h = tanh(a)
    return 1 - h*h


