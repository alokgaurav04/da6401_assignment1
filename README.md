# DA6402 Assignment1 :

[WANDB_Report](https://wandb.ai/alokgaurav04-indian-institute-of-technology-madras/DA6401_Assignment_1/reports/DA6401-Assignment-1--VmlldzoxMTYwOTMxNw)

Implementation of a Feed Forward Neural Network with various Optimization algorithms

#Question 1:

Run the Question_1.py file to get images in wandb.

#Question 2 :

Run the Q_No_2.py file to output the probability distribution for all classes of some samples in Fashion_Mnist data.
Comment the Line No-163 ( wandb.log() ) in NeuralNetwork.py while running this code 

#Question 3:

The backpropagation algorithm is implemented in NeuralNetwork.py file.

Defining the Neural Network for classification:
model = FeedForwardNN(input_size, hidden_layers, output_size,activation=config.activation_func,weight_init=config.weight_init,loss_func='cross_entropy', learn_rate=config.learning_rate, grad_desc=config.optimizer)

Training the network :
model.train(x_train, y_train,batch_size,epochs=10,val_split=0.1)  #Batch Size can be changed from this function

Adding a new optimisation alogorithm:
Add the code for new algorithm in "update_params(self)" function inside NeuralNetwork.py file

#Question 4:
sweep.py file for sweep functionality provided by wandb to find the best values for the hyperparameters

#Question 5:
Plot is provided in the report

#Question 6:  
"Parallel co-ordinates plot" , "correlation summary" and the observations provided in the report

#Question 7:
confusion_matrix.py file provided to plot the confusion matrix.
Code for finding the best model in a sweep using validation accuracy and test accuracy is provided in this python file.

#Question 8:
In sweep.py file :
model = FeedForwardNN(input_size, hidden_layers, output_size,activation=config.activation_func,weight_init=config.weight_init,loss_func='cross_entropy', learn_rate=config.learning_rate, grad_desc=config.optimizer)

Change loss_func='MSE' to find the mean square error instead of 'cross_entropy' .

#Question 10:
mnist.py file provided to run the code for MNIST dataset.


