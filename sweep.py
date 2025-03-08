from keras.datasets import fashion_mnist
import numpy as np
from NeuralNetwork import FeedForwardNN
from Loss_Function import *
import wandb
import os

os.environ['WANDB_TIMEOUT'] = '120'  # Set timeout to 60 seconds

def train_wandb(config = None):

    input_size = 28 * 28  # Flattened image size
    hidden_layers=[]
    no_of_hidden_layer=config.hidden_layers
    hidden_layer_size=config.hidden_layer_size
    for i in range(no_of_hidden_layer):
        hidden_layers.append(hidden_layer_size)   # eg :-  hidden_layers = [128, 128] , Number of neurons in each hidden layer
    
    output_size = 10  # Number of classes
    # Load the Fashion MNIST dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Preprocess the data
    X_train = X_train.reshape(-1, 28*28) / 255.0  # Flatten the images and normalize
    X_test = X_test.reshape(-1, 28*28) / 255.0

    # One-hot encode the labels
    y_train_one_hot = np.zeros((y_train.shape[0], 10))
    y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1

    y_test_one_hot = np.zeros((y_test.shape[0], 10))
    y_test_one_hot[np.arange(y_test.shape[0]), y_test] = 1

    run = wandb.init(config=config, resume=True)
    config = wandb.config

    name = f'bs_{config.batch_size}_acf_{config.activation_func}_lr_{config.learning_rate}_opt_{config.optimizer}_w_init_{config.weight_init}'
    wandb.run.name = name
    # wandb.run.save()

    model = FeedForwardNN(input_size, hidden_layers, output_size, learn_rate=config.learning_rate, grad_desc=config.optimizer)

    model.train(X_train, y_train_one_hot, batch_size=config.batch_size, epochs=config.epochs,val_split=0.1)

project_name = 'DA6401_Assignment_1' #Add project name here
entity = '' #Add username here
# wandb.init(project=project_name)

sweep_config = {
    'method': 'bayes', 
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'   
              },
    'parameters': {
        'epochs': {
            'values': [5,10]
        },
        'hidden_layers': {
            'values': [3,4,5]
        },
        'hidden_layer_size' : {
            'values' : [32,64,128,256]
        },
        'learning_rate': {
            'values': [0.001,0.0001]
        },
        'optimizer': {
            'values': ["sgd","momentum","nesterov", "rmsprop", "adam","nadam"]
        },
        'batch_size': {
            'values': [16,32,64,128]
        },
        'weight_init': {
            'values': ["random_init", "xavier_init"]
        },
        'activation_func': {
            'values': ["sigmoid","tanh","relu"]
        },
        'weight_decay': {
            'values': [0,0.0005,0.5]
        }
    }
}

#To add a new agent to an existing sweep, comment next line and directly put sweep_id in wandb.agent
sweep_id = wandb.sweep(sweep_config, project=project_name)

wandb.agent(sweep_id, project=project_name, function=train_wandb,count=1)