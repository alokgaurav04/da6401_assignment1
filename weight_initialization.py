import numpy as np
import math

def rand_initialization(prev_neurons,curr_neurons):
    weights = np.random.randn(prev_neurons,curr_neurons)*0.1
    bias=np.random.randn(1,curr_neurons)
    return weights,bias