import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist,mnist
from weight_initialization import *
from Loss_Function import *
from Activation import *
import wandb

class FeedForwardNN:
    def __init__(self,input_size,hidden_layers,output_size=10,activation='relu',weight_init='random_init',loss_func='cross_entropy',learn_rate=0.01,grad_desc='sgd',beta_1=0.9,beta_2=0.999,epsilon=1e-8,momentum=0.9,rho=0.9):
        self.hidden_layers=hidden_layers
        self.activation=activation
        self.weight_init=weight_init
        self.loss_func=loss_func
        self.learn_rate=learn_rate
        self.grad_desc=grad_desc
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.epsilon=epsilon
        self.momentum=momentum
        self.rho=rho

        #initialize layer
        self.weights=[]
        self.biases=[]
        layer_size=[input_size]+hidden_layers+[output_size]

        #initialise weights and biases
        for i in range(len(layer_size)-1):
            (weight,bias)= eval(self.weight_init+"(layer_size[i],layer_size[i+1])")
            self.weights.append(weight)
            self.biases.append(bias)
        
        #initialize momentum
        self.v_dw=[np.zeros_like(w) for w in self.weights]
        self.v_db=[np.zeros_like(b) for b in self.biases]
        self.s_dw=[np.zeros_like(w) for w in self.weights]
        self.s_db=[np.zeros_like(b) for b in self.biases]
        self.m_dw=[np.zeros_like(w) for w in self.weights]
        self.m_db=[np.zeros_like(b) for b in self.biases]
    
    def forward(self,x):
        #forward pass
        self.a=[] # activation function
        self.z=[] # output

        current_input=x
        for i in range(len(self.weights)-1):
            z=np.dot(current_input,self.weights[i])+self.biases[i] 
            a=eval(self.activation+"(z)")  #a=relu(z)
            self.z.append(z)
            self.a.append(a)
            current_input=a
        
        # output layer
        z=np.dot(current_input,self.weights[-1]) + self.biases[-1]
        self.z.append(z) 
        self.a.append(softmax(z))

        return self.a[-1]
    
    def backward(self,x,y,m):
        self.dw=[]
        self.db=[]

        #Error at the output layer
        if(self.loss_func=='cross_entropy'):
           dz=self.a[-1] - y 
        else:
           dz=MSE_Grad(y,self.a[-1])
        
        self.dw.append(np.dot(self.a[-2].T,dz))
        self.db.append(np.sum(dz,axis=0,keepdims=True))

        # Backpropagation through hidden layers
        for i in range(len(self.hidden_layers)-1,-1,-1):
            dz=np.dot(dz,self.weights[i+1].T)*eval(self.activation + "_derivative" + "(self.z[i])")  #relu_derivative(self.z[i])
            self.dw.insert(0,np.dot(self.a[i-1].T,dz) if i!=0 else np.dot(x.T,dz))
            self.db.insert(0,np.sum(dz,axis=0,keepdims=True))
        
    def update_params(self):
        # weights and biases update grad_descs
        if self.grad_desc=='sgd':
            for i in range(len(self.weights)):
                self.weights[i]=self.weights[i]-self.learn_rate*self.dw[i]
                self.biases[i]=self.biases[i]-self.learn_rate*self.db[i]
        elif self.grad_desc=='momentum':
            for i in range(len(self.weights)):
                self.v_dw[i]=self.v_dw[i]*self.momentum + self.learn_rate*self.dw[i]
                self.v_db[i]=self.v_db[i]*self.momentum + self.learn_rate*self.db[i]
                self.weights[i]=self.weights[i]-self.v_dw[i]
                self.biases[i]=self.biases[i]-self.v_db[i]
        elif self.grad_desc == 'nesterov':
            for i in range(len(self.weights)):
                v_dw_prev = self.v_dw[i].copy()
                v_db_prev = self.v_db[i].copy()
                self.v_dw[i] = self.momentum * self.v_dw[i] + self.learn_rate * self.dw[i]
                self.v_db[i] = self.momentum * self.v_db[i] + self.learn_rate * self.db[i]
                self.weights[i] -= self.momentum * v_dw_prev + (1 + self.momentum) * self.v_dw[i]
                self.biases[i] -= self.momentum * v_db_prev + (1 + self.momentum) * self.v_db[i]

        elif self.grad_desc == 'rmsprop':
            for i in range(len(self.weights)):
                self.s_dw[i] = self.rho * self.s_dw[i] + (1 - self.rho) * self.dw[i] ** 2
                self.s_db[i] = self.rho * self.s_db[i] + (1 - self.rho) * self.db[i] ** 2
                self.weights[i] -= self.learn_rate * self.dw[i] / (np.sqrt(self.s_dw[i]) + self.epsilon)
                self.biases[i] -= self.learn_rate * self.db[i] / (np.sqrt(self.s_db[i]) + self.epsilon)
        
        elif self.grad_desc == 'adam':
            for i in range(len(self.weights)):
                self.m_dw[i] = self.beta_1 * self.m_dw[i] + (1 - self.beta_1) * self.dw[i]
                self.m_db[i] = self.beta_1 * self.m_db[i] + (1 - self.beta_1) * self.db[i]
                self.v_dw[i] = self.beta_2 * self.v_dw[i] + (1 - self.beta_2) * (self.dw[i] ** 2)
                self.v_db[i] = self.beta_2 * self.v_db[i] + (1 - self.beta_2) * (self.db[i] ** 2)

                m_dw_hat = self.m_dw[i] / (1 - self.beta_1)
                m_db_hat = self.m_db[i] / (1 - self.beta_1)
                v_dw_hat = self.v_dw[i] / (1 - self.beta_2)
                v_db_hat = self.v_db[i] / (1 - self.beta_2)

                self.weights[i] -= self.learn_rate * m_dw_hat / (np.sqrt(v_dw_hat) + self.epsilon)
                self.biases[i] -= self.learn_rate * m_db_hat / (np.sqrt(v_db_hat) + self.epsilon)

    def train(self, x_train, y_train,batch_size,epochs=10,val_split=0.1):
        no_of_examples=x_train.shape[0]
        val_size=int(no_of_examples*val_split)
        x_val=x_train[:val_size]
        y_val=y_train[:val_size]
        x_train=x_train[val_size:]
        y_train=y_train[val_size:]

        for epoch in range(epochs):
            for i in range(0,no_of_examples,batch_size):
                end_index=min(i+batch_size,no_of_examples)
                x_train_batch=x_train[i:end_index]
                y_train_batch=y_train[i:end_index]

                if x_train_batch.shape[0] > 0:
                   # Forward pass on the dataset
                   self.forward(x_train_batch)
                   # Backpropagation on the dataset
                   self.backward(x_train_batch, y_train_batch,x_train_batch.shape[0])
            
                   # Update parameters after computing gradients for the dataset
                   self.update_params()
            
            # Calculate loss after every epoch
            y_pred = self.predict(x_train)
            y_val_pred=self.predict(x_val)

            train_loss=eval(self.loss_func+"(y_train,y_pred)") ##cross_entropy(y_train,y_pred)
            
            train_accuracy=accuracy_function(y_train,y_pred)   


            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss},Accuracy :{train_accuracy}")

            if x_val is not None and y_val is not None:
                val_loss=eval(self.loss_func+"(y_train,y_pred)")
                val_accuracy=accuracy_function(y_val,y_val_pred)
                print(f"Epoch {epoch+1}/{epochs} Validation Loss: {val_loss}, validation Accuracy :{val_accuracy}")
            wandb.log({"val_loss": val_loss,"val_accuracy": val_accuracy,"train_loss": train_loss,"train_accuracy": train_accuracy})

    def predict(self, X):
        prob = self.forward(X)
        return prob  # Return the probability distribution over the 10 classes


  


        