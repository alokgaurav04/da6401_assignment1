# DA6402 Assignment1 :

[WANDB_Report](https://wandb.ai/pranjalchitale/CS6910_A1/reports/CS6910-Assignment-1-FFN--VmlldzoxNTk5OTcx)

Implementation of a Feed Forward Neural Network with various Optimization algorithms

Question 1:

Run the Question_1.py file to get images in wandb.

Question 2 :

Run the Q_No_2.py file to output the probability distribution for all classes of some samples in Fashion_Mnist data.

Question 3:

The backpropagation algorithm is implemented in NeuralNetwork.py file.

## Defining the Neural Network for classification:
nn = FFN(input_dim = 784, num_classes = 10, optimizer=config.optimizer, weight_decay=config.weight_decay, loss_fn= config.loss_fn)
