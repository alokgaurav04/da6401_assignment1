import wandb
from keras.datasets import fashion_mnist 
import numpy as np 

run =wandb.init(project="DA6401_Assignment_1")
 
fmnist_labels = { 0 : 'T-shirt/top', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt', 7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle boot'} 
 
 #load the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()  

samples = [] 
#iterate over all unique classes
for i in np.unique(y_train): 
    index = np.where(y_train == i)[0][0] # Index of the 1st image of each class
    pixels = x_train[index]  # get the image
    image = wandb.Image(pixels, caption=f"{fmnist_labels[y_train[index]]}") 
    samples.append(image) 

run.log({"Samples Images for each class": samples})
