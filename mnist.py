from NeuralNetwork import FeedForwardNN,mnist,np
import matplotlib.pyplot as plt
from Loss_Function import *
import wandb

# Fetch the best model from the sweep
api = wandb.Api()
sweep = api.sweep(f"DA6401_Assignment_1/kvo0qk0z")

# Sort runs by validation accuracy
best_run = sorted(sweep.runs, key=lambda run: run.summary.get("val_accuracy", 0), reverse=True)[2]

# Extract best model's hyperparameters
best_config = best_run.config
print(f"Best model found: {best_config}")

input_size = 28 * 28  # Flattened image size
output_size = 10  # Number of classes
hidden_layers=[]
no_of_hidden_layer=best_config['hidden_layers']
hidden_layer_size=best_config['hidden_layer_size']
for i in range(no_of_hidden_layer):
    hidden_layers.append(hidden_layer_size)

# Initialize the neural network
nn = FeedForwardNN(input_size, hidden_layers, output_size,activation=best_config['activation_func'],weight_init=best_config['weight_init'],loss_func='MSE', learn_rate=best_config['learning_rate'], grad_desc=best_config['optimizer'])

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
nn.train(X_train, y_train_one_hot,batch_size=best_config['batch_size'], epochs=best_config['epochs'],val_split=0.1)

# # Test the model
y_pred = nn.predict(X_test)
test_accuracy=accuracy_function(y_test_one_hot,y_pred)
print('Test Accuracy :',test_accuracy)

