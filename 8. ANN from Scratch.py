# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:13:56 2020

@author: LENOVO
"""

# Import libraries
import numpy as np
import pylab as plt
import time


"Multilayer Perceptron"


# Training Data (multiple samples)
# BEGIN user input
inputData = np.array([[0.0], [0.2], [0.5], [0.7], [0.85]])     # Input layer, arbitrary number of neurons
# =============================================================================
# arbitrary number of hidden layers with arbitrary number of neurons within
# =============================================================================
outputData = np.array([[0], [0.5], [0.6], [0.75], [0.8]])    # Output layer, arbitrary number of neurons
# END user input


# Class for batch training -> arbitrary number of layers, neurons, samples and arbitrary activation function
class Neural_Network(object):
    def __init__(self, layers=[], activationFunction=None):
        '''List layers is a list of integers where every integer represents the number of neurons in the layer.
        The first integer represents the number of neurons in the INPUT layer and the last one number of neurons in the OUTPUT layer.
        Integers in-between the first and the last one represent the number of neurons in hidden layers if there are.
        The second integer in the list represents the neurons of the first hidden layer and penultimate integer the neurons
        in the last hidden layer respectively.
        Default activationFunction is linear. Also, others could be used: activationFunction = "sigmoid" or "ReLU" or "TanH"'''
        self.layers = layers  # list of layers
        self.activationFunction = activationFunction

        # Create a list of weight matrices from the “standard normal” distribution
        self.W = []
        for i in range(len(self.layers)-1):
            self.W.append(np.random.randn(self.layers[i], self.layers[i+1]))
            
        # Biases (random starting parameters)
        self.B = []
        for i in range(len(self.layers)-1):
            self.B.append(np.random.randn(1, self.layers[i+1]))
                
    def activation(self, z):
        "Define activation function"
        if self.activationFunction == "Sigmoid":
            return 1/(1 + np.exp(-z))
        elif self.activationFunction == "ReLU":
            m = z.copy()        # use m as a copy of the input z to avoid overwriting of the original input
            for cell in np.nditer(m, op_flags=['readwrite']):   # np.nditer - iterating over an array(matrix)
                if cell[...] < 0:
                    cell[...] = 0
            return m
        elif self.activationFunction == "TanH":
            return 2/(1 + np.exp(-2*z)) - 1
        else:
            return z  # linear activation function by default
        
    def activationPrime(self, z):
        "Define activation function derivative"
        if self.activationFunction == "Sigmoid":
            return np.exp(-z)/((1 + np.exp(-z))**2)
        elif self.activationFunction == "ReLU":
            n = z.copy()
            for cell in np.nditer(n, op_flags=['readwrite']):
                if cell[...] < 0:
                    cell[...] = 0
                else:
                    cell[...] = 1
            return n
        elif self.activationFunction == "TanH":
            return 1 - (2/(1 + np.exp(-2*z)) - 1)**2
        else:
            return 1  # linear activation function by default      
                    
    def forward(self, X):       # actication function applied to the input layer
        "Propagate inputs through the network"
        self.Z = [np.dot(X, self.W[0]) + self.B[0]]  # list of neuron z values (the first one is diferent from template)
        self.A = [self.activation(self.Z[0])]     # list of activated neurons
        for i in range(len(self.layers)-2):       # -1 for the input layer and -1 for the first hidden layer already added
            self.Z.append(np.dot(self.A[i], self.W[i+1])+self.B[i+1])    # apply weights W, add biases B to the neurons
            self.A.append(self.activation(self.Z[i+1]))                 # apply activationFunction
        return self.A[-1]       # predicted output
   
    def costFunction(self, y):
        "returns cost function value for multiple outputs"
        J = 0.5*((self.A[-1] - y)**2)
        return J.sum()  # sum of the elements in matrix J      
    
    def costFunctionPrime(self, X, y):       # actication function applied to the input layer
        "Compute derivatives with respect to the weights W and Biases B"
        self.delta = [(self.A[-1] - y)*self.activationPrime(self.Z[-1])]   # list of delta values (the last one is diferent from template), * Hadamard product
        self.dJdW = []                          # list of derivatives w.r.t. the weights
        for i in range(1, len(self.layers)-1):   # range from 1 enables insert -1(last) element from a list
            self.delta.insert(0, (np.dot(self.delta[-i], self.W[-i].T))*self.activationPrime(self.Z[-i-1]))  # insert element to the first place in the list self.delta
            self.dJdW.insert(0, np.dot(self.A[-i-1].T, self.delta[-i]))   # insert element to the first place in the list self.dJdW
        self.dJdW.insert(0, np.dot(X.T, self.delta[0]))  # the first one is diferent from template
        self.dJdB = []
        for i in range(len(self.delta)):
            self.dJdB.append(self.delta[i].mean(axis=0))  # mean value of gradient per samples (axis=0 -> mean value per columns of the matrix where one sample is one row()
        return self.dJdW
          
    def getW(self):
        "return list of weights merged into single array"
        # ravel converts matrix to array, np.concatenate creates a single array
        weights = np.concatenate([i.ravel() for i in self.W])  # par -> parameter
        return weights
    
    def setW(self, weights):
        "set new weights from a single weights vector"
        boundary = [0]  # boundaries for cutting matrices from a singe vector
        for i in range(len(self.layers)-1):
            boundary.append(boundary[i] + self.layers[i]*self.layers[i+1])
            self.W[i] = np.reshape(weights[boundary[i]:boundary[i+1]], (self.layers[i], self.layers[i+1]))
            # the firt part in () inserts elements inside boundaries
            # the second one defines matrix dim for inserted elements
            
    def getB(self):
        "return list of biases merged into single array"
        # ravel converts matrix to array, np.concatenate creates a single array
        biases = np.concatenate([i.ravel() for i in self.B])  # par -> parameter
        return biases
    
    def setB(self, biases):
        "set new biases from a single biases vector"
        boundary = [0]  # boundaries for cutting matrices from a single vector
        for i in range(len(self.layers)-1):
            boundary.append(boundary[i] + self.layers[i+1])
            self.B[i] = np.reshape(biases[boundary[i]:boundary[i+1]], (1, self.layers[i+1]))
            # the firt part in () inserts elements inside boundaries
            # the second one defines matrix dim for inserted elements
                                      
    def gradW(self, X, y):
        "return partial derivatives w.r.t. the weights merged into single array(weight gradient)"
        dJdW = self.costFunctionPrime(X,y)
        return (1/batchSize)*np.concatenate([i.ravel() for i in dJdW])  # mean value of gradient per samples
    
    def gradB(self):
        "return partial derivatives w.r.t. the biases merged into single array(bias gradient)"
        return np.concatenate([i.ravel() for i in self.dJdB])

    def trainNetwork(self, alpha, maxIteration):    # using Batch Gradient Descent (BGD)
        '''alpha represents learning rate in range [0,1] and
        maxIteration represents maximal number of iterations'''
        self.alpha = alpha      # learning rate(from 0.0 to 1.0)
        self.maxIteration = maxIteration    # arbitrary 
        for i in range(self.maxIteration):
            self.forward(X)
            currentW = self.getW()  # weights before applying GD
            currentB = self.getB()  # biases before applying GD
            newW = currentW - self.alpha * self.gradW(X, y)   # apply GD to get new weights
            newB = currentB - self.alpha * self.gradB()       # apply GD to get new biases
            self.setW(newW)         # set weights for the next iteration
            self.setB(newB)         # set biases fot the next iteration

           
          

startTime = time.process_time()

# general training algorythm (BGD, MBGD, SGD) using predifined class Neural_Network
print("\n<< Neural Network Training >>\n")
# empty lists for plot will be eppended during training
epochPlot = [0]
costFunctionPlot = []  # cost function is plotted after every epoch
epochNumber = 200  # user input -> choose number of epochs
if type(epochNumber) != int:
    raise ValueError("epochNumber must be an integer")
batchSize = 5   # user input -> any divisor of len(inputData)
batchNumber = int(len(inputData)/batchSize)
if len(inputData) % batchSize != 0:
    raise ValueError("batchSize must be a factor of len(inputData)")    
for epoch in range(epochNumber):
    epochPlot.append(epoch + 1)
    for batch in range(batchNumber):
        X = inputData[batch*batchSize:batch*batchSize + batchSize]      # cut batch from input data
        y = outputData[batch*batchSize:batch*batchSize + batchSize]     # cut batch from output data
        if batch == 0 and epoch == 0:      # assign initial weigts and biases only once
            # BEGIN user input
            NN = Neural_Network([1, 5, 4, 3, 2, 1], activationFunction="TanH")       # user input -> define arhitecture of NN and choose an activation function
            NN.setW([0.2, 0.3, 0.1, 0.4, 0.2,
                     0.4, 0.45, 0.1, 0.4, 0.5, 0.15, 0.1, 0.4, 0.25, 0.2, 0.25, 0.4, 0.2, 0.15, 0.3, 0.1, 0.4, 0.45, 0.1, 0.5,
                     0.1, 0.3, 0.25, 0.1, 0.1, 0.5, 0.2, 0.4, 0.5, 0.1, 0.3, 0.3,
                     0.3, 0.3, 0.1, 0.2, 0.4, 0.1,
                     0.5, 0.2])     # user (arbitrary)input -> set specific initial weights
            NN.setB([0.1, 0.1, 0.3, 0.1, 0.4,
                     0.1, 0.3, 0.1, 0.3,
                     0.15, 0.1, 0.4,
                     0.2, 0.1,
                     0.2])      # user (arbitrary)input -> set specific initial biases
            costFunctionPlot.append(((0.5/len(inputData))*((NN.forward(inputData)-outputData)**2)).sum())   # cost function for initial parameters and epoch = 0
        NN.trainNetwork(0.3, 1)             # single batch training (learning rate(user input), 1(default))
        # END user input
    costFunctionPlot.append(((0.5/len(inputData))*((NN.forward(inputData)-outputData)**2)).sum())   # cost function after every epoch for inputData
print("Predicted Training Values: \n", NN.forward(inputData), "\n")
print("Real Training Values: \n", outputData, "\n")
print("Activation Function: ", NN.activationFunction)
print("Epoch Number: ", epochNumber)
print("Learning Rate: ", NN.alpha)
print("Cost Function: ", ((0.5/len(inputData))*((NN.forward(inputData)-outputData)**2)).sum())
# Plot Setup   
plt.figure("Cost")
plt.plot(epochPlot, costFunctionPlot)
plt.xlabel("Epochs")
plt.ylabel("Cost Function")
plt.grid(True)
plt.figure("Output")
plt.plot(list(range(1, outputData.size + 1)), NN.forward(inputData), 'ro', label='Predicted')  # 'ro' - red dot
plt.plot(list(range(1, outputData.size + 1)), outputData, 'bo', label='Real')  # 'bo' - blue dot, y.size - number of the elements in matrix y
plt.xlabel("Output Element")
plt.ylabel("Output Value")
plt.legend()
plt.grid(True)

endTime = time.process_time()
print("Training time: ", endTime - startTime)















