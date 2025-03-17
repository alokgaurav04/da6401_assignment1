from NeuralNetwork import FeedForwardNN,fashion_mnist,np
import matplotlib.pyplot as plt
from Loss_Function import *

# Example Usage
input_size = 28 * 28  # Flattened image size
#hidden_layers = [128, 128]  # Number of neurons in each hidden layer
output_size = 10  # Number of classes
hidden_layers=[]
no_of_hidden_layer=2
hidden_layer_size=128
for i in range(no_of_hidden_layer):
    hidden_layers.append(hidden_layer_size)

# Initialize the neural network
nn = FeedForwardNN(input_size, hidden_layers, output_size,activation='relu',weight_init='random_init',loss_func='MSE', learn_rate=0.001, grad_desc='adam')

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

# Train the model
nn.train(X_train, y_train_one_hot,batch_size=40, epochs=5,val_split=0.1)

# # Test the model
y_pred_prob = nn.predict(X_test)

# Print the probability distribution for the first 5 test samples
for i in range(5):
    print(f"Sample {i+1} probability distribution:")
    print(np.round(y_pred_prob[i],2))
    print("\n")

y_pred_class = np.argmax(y_pred_prob, axis=1)
print(f"Predicted class for first 5 samples: {y_pred_class[:5]}")