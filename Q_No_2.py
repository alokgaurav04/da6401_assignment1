from NeuralNetwork import FeedForwardNN,fashion_mnist,np

# Example Usage
input_size = 28 * 28  # Flattened image size
hidden_layers = [128, 128]  # Number of neurons in each hidden layer
output_size = 10  # Number of classes

# Initialize the neural network
nn = FeedForwardNN(input_size, hidden_layers, output_size, learn_rate=0.001, grad_desc='adam')

# Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 28*28) / 255.0  # Flatten the images and normalize

#Forward propagation in Feed Forward Neural Network
prob_dist=nn.forward(X_train)

for i in range(5):
    print(f"Sample {i+1} probability distribution:")
    print(prob_dist[i])
    print("\n")
