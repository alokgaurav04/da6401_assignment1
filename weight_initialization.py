import numpy as np
import math

def random_init(prev_neurons,curr_neurons):
    weights = np.random.randn(prev_neurons,curr_neurons)*0.1
    bias=np.random.randn(1,curr_neurons)
    return weights,bias

def xavier_init(prev_neurons, curr_neurons):
    lower_limit, upper_limit = -math.sqrt(6.0/(curr_neurons + prev_neurons)), math.sqrt(6.0/(curr_neurons + prev_neurons)) 
    weights = np.random.uniform(lower_limit, upper_limit, size=(prev_neurons, curr_neurons))
    bias = np.random.uniform(lower_limit, upper_limit, size=(1, curr_neurons))
    return weights, bias