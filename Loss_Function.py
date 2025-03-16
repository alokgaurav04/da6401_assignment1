import numpy as np
#Loss Functions
def cross_entropy(y_train, y_pred): 
    loss=-np.mean(np.sum(y_train * np.log(y_pred + 1e-9), axis=1))
    return loss

def accuracy_function(y_true,y_pred):
    return np.mean(np.argmax(y_pred,axis=1)==np.argmax(y_true,axis=1))

def MSE(y_true, y_pred):
    loss = np.mean(np.square(y_pred - y_true))
    return loss

def MSE_Grad(y_true, y_pred):
    diff = y_pred - y_true
    #temp = np.multiply(y_true, y_pred).sum(axis = 1, keepdims = True)
    return diff/(y_pred.shape[0])