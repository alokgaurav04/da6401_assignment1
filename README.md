# DA6402 Assignment1 :

[WANDB_Report](https://wandb.ai/alokgaurav04-indian-institute-of-technology-madras/DA6401_Assignment_1/reports/DA6401-Assignment-1--VmlldzoxMTYwOTMxNw)

Implementation of a Feed Forward Neural Network with various Optimization algorithms

#Question 1:

Run the Question_1.py file to get images in wandb.

#Question 2 :

Run the Q_No_2.py file to output the probability distribution for all classes of some samples in Fashion_Mnist data.

#Question 3:

The backpropagation algorithm is implemented in NeuralNetwork.py file.

Defining the Neural Network for classification:
nn = FFN(input_dim = 784, num_classes = 10, optimizer=config.optimizer, weight_decay=config.weight_decay, loss_fn= config.loss_fn)

Training the network :
nn.train(x_train, y_train,batch_size,epochs=10,val_split=0.1)  #Batch Size can be changed from this function

Adding a new optimisation alogorithm:
Add the code for new algorithm in "update_params(self)" function inside NeuralNetwork.py file

#Question 4:

