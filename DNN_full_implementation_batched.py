# -*- coding: utf-8 -*-
"""
This trains a MLP for any classification problem where the input feature vector is relatively low-dimensional. Code contains example to train MLP on the **Red Wine Classification** dataset for binary-classification.
Performs Stochastic Gradient Descent over a batched input.

This is how to define and train it.

myMLP = NN()
num_epochs= 30
lambda_reg= 0.01
learning_rate= 0.001
batch_size= 32

myMLP.add_layer('Hidden', 11, 16) 
myMLP.add_layer('Hidden', 16, 12)
myMLP.add_layer('Hidden', 12, 8)
myMLP.add_layer('Output', 8, 4)
myMLP.add_layer('Loss', 4, 2)
"""

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

class Base:
    def __init__(self, input_dims:int, output_dims:int):
        self.input_dims = input_dims
        self.output_dims = output_dims

    def forward_pass(self):
        pass 
    
    def backward_pass(self):
        pass

    def update_weights(self, W, b, del_W, del_b, learning_rate):
        W-=(learning_rate*del_W)
        b-=(learning_rate*del_b)
        return W, b

class Hidden(Base):
    def __init__(self, input_dims:int, output_dims:int):
        super().__init__(input_dims, output_dims)
        
        self.W = np.random.random((input_dims, output_dims)) - 0.5
        self.c = np.random.random((1, output_dims)) - 0.5

    def forward_pass(self, X):
        U =X@ self.W + self.c
        activations= self.leaky_relu(U)
        return activations

    def backward_pass(self, X, h, dLdh, alpha, learning_rate):
        relu_derivative= self.leaky_relu_derivative(h)
        dLdU = np.multiply(relu_derivative, dLdh)
        dLdW = X.T@ dLdU + alpha*self.W
        dLdc = dLdU.mean(axis=0, keepdims=True)   # as bias is broadcast over the batch, take mean of gradients of batch
        for_prev = dLdU@ self.W.T                                       

        self.W, self.c = self.update_weights(self.W, self.c, dLdW, dLdc, learning_rate)
        return for_prev #"""Note: for_prev is the gradient dL/dh' we pass onto the previous layer h' """ 

    def leaky_relu(self, inp):
      activation_mask = 1.0 * (inp >0) + 0.01*(inp<0)
      activations= np.multiply(inp, activation_mask)
      return activations 

    def leaky_relu_derivative(self, h):
      """
      Returns a numpy ndarray of shape (batch_size x self.output_dims)
      """
      leaky_relu_derivative = 1.0 * (h >0) + 0.01*(h<=0)
      return leaky_relu_derivative

class Hidden_Vondrick(Base):
    def __init__(self, input_dims:int, output_dims:int):
        super().__init__(input_dims, output_dims)
        
        self.W = np.random.random((input_dims, output_dims))
        self.c = np.random.random((1, output_dims))
        self.U= None
        self.vondrick_exponent= np.random.uniform(1.4,2) #The learnable exponent, called Vondrick Exponent, for our custom activation function is initilized from a Unifom(1.4,2) distribution. May change this, but keep it close to this range to ensure training stability.

        print("Intital Value of Vondrick Exponent: "+ str(self.vondrick_exponent) )

    def forward_pass(self, X):
        self.U =X@ self.W + self.c
        activations= self.vondrick_activation(self.U)
        return activations

    def backward_pass(self, X, h, dLdh, alpha, learning_rate=0.0005):
        relu_derivative, exponent_derivative = self.vondrick_activation_derivative()
        dLdU = np.multiply(relu_derivative, dLdh)
        dLdW = X.T@ dLdU + alpha*self.W
        dLdc = dLdU.mean(axis=0, keepdims=True)   # as bias is broadcast over the batch, take mean of gradients of batch
        dL_dexponent_scalar = np.multiply(exponent_derivative, dLdh).mean()   # as expontnt is broadcast over the batch and over each dimension, take overall mean to get a scalar
        for_prev = dLdU@ self.W.T

        #Note, that for the purposes of training stablity, we have hard-coded the learning rate here to be 0.0005
        self.W, self.c = self.update_weights(self.W, self.c, dLdW, dLdc, 0.0005) 

        #Gradient Descent on our learnable activation function parameter: Updating exponent, but clipping it's lower range to 1.01
        self.vondrick_exponent= max(1.01,self.vondrick_exponent- 0.001*dL_dexponent_scalar )

        return for_prev

    def vondrick_activation(self, U):
      U_pos= np.maximum(U,0)
      activations= np.power(U_pos, self.vondrick_exponent) + np.multiply(U, 0.01*(U<0) )
      return activations 



    def vondrick_activation_derivative(self):
      """
      TO DO:
      Uses the stored self.U to do a backward pass and return both dh/dU and dh/dexponent. Both are numpy matrices dimensions batch_size x self.output_dims 
      return activations: (batch_size x self.output_dims)
      """
      U_pos = np.maximum(self.U,0)
      # dh/dU = {vondrick_exponent * u^(vondrick_exponent-1),     when u>0
      #          0.01,                                            when u<=0 }
      derivative_wrt_U = ( self.vondrick_exponent * np.power(U_pos, (self.vondrick_exponent -1)) ) + 0.01*(self.U<=0)
      # dh/dexponent = {ln(u) * u^vondrick_exponent,     when u>0
      #                 0,                               when u<=0 }
      derivative_wrt_exponent = np.multiply( np.log(U_pos) , np.power(U_pos, self.vondrick_exponent) ) + 0*(self.U<=0)

      np.nan_to_num(derivative_wrt_U, copy=False)
      np.nan_to_num(derivative_wrt_exponent, copy=False)
      
      return derivative_wrt_U, derivative_wrt_exponent

    
class Output(Base):

    def __init__(self, input_dims, output_dims):
        super().__init__(input_dims, output_dims)
        self.w = np.random.random((input_dims, output_dims)) -0.5
        self.b = np.random.random((1, output_dims)) -0.5

    def forward_pass(self, h):
        z = h @self.w + self.b
        z = z - np.max(z, axis = 1).reshape(z.shape[0], 1) # trick: subtracting maz z as softmax is not effected: prevents overflow when we do exponentation
        return z

    def backward_pass(self, h, dLdz, alpha, learning_rate):
        dLdw = h.T@ dLdz + alpha*self.w
        dLdb = dLdz.mean(axis=0, keepdims=True)   # as bias is broadcast over the batch, take mean of gradients of batch
        dLdh = dLdz@ self.w.T


        self.w, self.b = self.update_weights(self.w, self.b, dLdw, dLdb, learning_rate)
        return dLdh

class Loss(Base): 
    
    def __init__(self, input_dims, output_dims):
        super().__init__(input_dims, output_dims)
    
    def forward_pass(self, z, y):

        temp = -z + np.log(np.sum(np.exp(z), axis = 1)).reshape(z.shape[0], 1) #Computing Softmax Cross Entropy Loss terms for each z_i. Note dimensions of temp: batch_size x output layer output_dims
        L = temp[np.arange(z.shape[0]), y.flatten().astype(int)] #Extracts Loss term corresponding only to ground truth class from each row (sample). 
        L = np.mean(L) #Mean Loss over the batch
        return L 

    def backward_pass(self, z, y):
        #Recall the simplified expression we get for dL_i/dz_k= p_k- I(y_i=k) (Details in the guide)
        temp1 = np.zeros(z.shape)
        for i in range(z.shape[0]):
            true_class = int(y[i].item())
            temp1[i][true_class] = -1     #-1 is added to the loss term corresponding to the true class

        temp2 = np.exp(z)/ np.sum(np.exp(z), axis =1 ).reshape(z.shape[0], 1) #Matrik of p_k terms, aka, elements replaced by softmaxed probabilities
        for_previous = temp1 + temp2 
        return for_previous
    
class NN: 
    def __init__(self): 
        self.output_layer= self.loss_layer =  None 
        self.hidden_layers = []

    def add_layer(self, name, input_dims, output_dims):
        if name.lower() == 'hidden':
            self.hidden_layers.append(Hidden(input_dims, output_dims))
        elif name.lower() == 'hidden_vondrick':
            self.hidden_layers.append(Hidden_Vondrick(input_dims, output_dims))
        elif name.lower() =='output':
            self.output_layer = Output(input_dims, output_dims)
        elif name.lower() =='loss':
            self.loss_layer = Loss(input_dims, output_dims) 
    
    def forward_prop(self, X, y, alpha): 
        hidden_outputs = []
        z = L = h= None 
        for layer in self.hidden_layers:
            h = layer.forward_pass(X)
            hidden_outputs.append(h)
            X = h 
        z = self.output_layer.forward_pass(h)
        L = self.loss_layer.forward_pass(z, y)
        for layer in self.hidden_layers:
            L += 0.5*alpha*np.linalg.norm(layer.W)**2
        L+= 0.5*alpha* np.linalg.norm(self.output_layer.w)**2 
        return hidden_outputs, z, L 

    def backward_prop(self, X, hidden_outputs, z, y, alpha =0.01, learning_rate =0.01):
        dLdz = self.loss_layer.backward_pass(z, y)
        for_previous = self.output_layer.backward_pass(hidden_outputs[-1], dLdz, alpha, learning_rate) 
        for i in range(len(self.hidden_layers)-1,0,-1):
            temp = self.hidden_layers[i].backward_pass(hidden_outputs[i-1], hidden_outputs[i], for_previous, alpha, learning_rate)
            for_previous = temp 
        self.hidden_layers[0].backward_pass(X, hidden_outputs[0], for_previous, alpha, learning_rate)

    def train(self, X, y, epochs, batch_size, learning_rate, alpha, show_training_accuracy=True): 
  
        loss = []
        for epoch in range(epochs):
            predicted = self.predict(X)
            correct = 0 
            for i in range(len(predicted)):
                if predicted[i] == y[i]:
                    correct+=1
            if show_training_accuracy:
                print(f'the accuracy on the training data after epoch {epoch + 1} is {correct/X.shape[0]}')
            temp = total = 0 
            for k in range(0, X.shape[0], batch_size):
                inp = X[k:k+batch_size]
                out = y[k:k+batch_size]

                hidden_outputs, z, L = self.forward_prop(inp, out, alpha)
                temp+=L 
                total+=1
                self.backward_prop(inp, hidden_outputs, z, out, alpha, learning_rate)
            
            loss.append(temp/total)

        return loss

    def predict(self, X): 
        """
        Takes in a batch input X: (number_of_samples x feature_vector_dims) and returns an np.array y of predictions (number_of_samples x 1)

        """
        h= None 
        for layer in self.hidden_layers:
            h = layer.forward_pass(X)
            X = h 
        output = self.output_layer.forward_pass(h)
        logits = np.exp(output - np.max(output)) / np.sum(np.exp(output - np.max(output)), axis=-1)[:,None]
        predictions = np.argmax(logits, axis=-1)
        return predictions
    
    def compute_accuracy(self, X, Y):
        predicted_Y= self.predict(X)
        correct=0
        for i in range(len(predicted_Y)):
            if predicted_Y[i] == Y[i]:
                  correct+=1
        return correct/len(Y)

def plot_loss(loss):
    #Given a list of losses over the epochs, plots the loss curve.
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss of the neural network per epoch")
    plt.plot(loss)
    plt.show()

"""# Testing MLP: Red Wine Quality Classification Dataset

## More about the dataset:
https://archive.ics.uci.edu/ml/datasets/wine+quality
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

"""Load the file winequality-red.csv"""
wine_dataset = pd.read_csv('./winequality-red.csv')

#Converting Labels to a Binary Classification Problem
def Convert_Labels(data):
    data.loc[:,'quality'] = np.where(data.loc[:,'quality']>=6, 1, 0)
    return data

#Scales features to constrain them to lie within the default range (0,1)
def DataScaler(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data

all_columns = list(wine_dataset)
target = ['quality']
print(all_columns)
features = list(set(all_columns)-set(target))
print(features)
wine_dataset.loc[:,features] = DataScaler(wine_dataset.loc[:,features])

wine_dataset.head()

label_converted_dataset = Convert_Labels(wine_dataset)

print(label_converted_dataset)
#As you can see, the quality column (our labels) now has either 0 (for quality<6) and 1 (for quality>=6)

#Quick Sanity check that our dataset it indeed relatively balanced
label_converted_dataset['quality'].mean()

y_wine = label_converted_dataset.loc[:,'quality']
X_wine = label_converted_dataset.drop(target,axis=1)

X_wine_np= np.asarray(X_wine)
y_wine_np= np.asarray(y_wine)

X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(X_wine_np, y_wine_np, test_size=0.25, random_state=1)

"""### Instantiating and Training our Red Wine Classifier MLP

First, we instantiate and train a standard MLP (that uses the RELU activation function).
"""

my_wine_NN_1 = NN()
num_epochs= 80
lambda_reg= 0.01
learning_rate= 0.001
batch_size= 32

my_wine_NN_1.add_layer('Hidden', 11, 16) #Note that the first layer's weight matrix must be 11 x k , as the input feature vector is 11-dimensional.
my_wine_NN_1.add_layer('Hidden', 16, 12)
my_wine_NN_1.add_layer('Hidden', 12, 8)
my_wine_NN_1.add_layer('Output', 8, 2)
my_wine_NN_1.add_layer('Loss', 4, 2)

loss_wine_li_1= my_wine_NN_1.train(X_wine_train, y_wine_train, num_epochs, batch_size, lambda_reg, learning_rate)
plot_loss(loss_wine_li_1)

print(my_wine_NN_1.compute_accuracy(X_wine_test, y_wine_test))

"""Now, we will include 1 "Hidden_Vondrick" Layer (i.e. a layer that leverages our custom activation function) in our MLP architecure."""

my_wine_NN_2 = NN()
num_epochs= 80
lambda_reg= 0.01
learning_rate= 0.001
batch_size= 32

my_wine_NN_2.add_layer('Hidden', 11, 16) #Note that the first layer's weight matrix must be 11 x k , as the input feature vector is 11-dimensional.
my_wine_NN_2.add_layer('Hidden', 16, 12)
my_wine_NN_2.add_layer('Hidden_Vondrick', 12, 8)
my_wine_NN_2.add_layer('Output', 8, 2)
my_wine_NN_2.add_layer('Loss', 4, 2)

loss_wine_li_2= my_wine_NN_2.train(X_wine_train, y_wine_train, num_epochs, batch_size, lambda_reg, learning_rate)
plot_loss(loss_wine_li_2)

print(my_wine_NN_2.compute_accuracy(X_wine_test, y_wine_test))
