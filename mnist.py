from NeuralNetwork import FeedForwardNN,mnist,np
import matplotlib.pyplot as plt
from Loss_Function import *

input_size = 28 * 28  # Flattened image size
output_size = 10  # Number of classes
hidden_layers=[]
no_of_hidden_layer=2
hidden_layer_size=128
for i in range(no_of_hidden_layer):
    hidden_layers.append(hidden_layer_size)

# Initialize the neural network
nn = FeedForwardNN(input_size, hidden_layers, output_size,activation='relu',weight_init='random_init',loss_func='MSE', learn_rate=0.001, grad_desc='adam')

# Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 28*28) / 255.0  # Flatten the images and normalize
X_test = X_test.reshape(-1, 28*28) / 255.0

# One-hot encode the labels
y_train_one_hot = np.zeros((y_train.shape[0], 10))
y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1

y_test_one_hot = np.zeros((y_test.shape[0], 10))
y_test_one_hot[np.arange(y_test.shape[0]), y_test] = 1

# Train the model
nn.train(X_train, y_train_one_hot,batch_size=40, epochs=5,val_split=0.1)

# # Test the model
y_pred = nn.predict(X_test)
test_accuracy=accuracy_function(y_test_one_hot,y_pred)
print('Test Accuracy :',test_accuracy)

