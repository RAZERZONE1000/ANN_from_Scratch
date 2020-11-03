# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:13:56 2020

@author: LENOVO
"""

# Import libraries
import numpy as np
import pylab as plt
import time


'''Example 6: multiple inputs - multiple neurons hidden layers - multiple outputs, 
              introducing biases and activation functions'''

# Training Data (1 sample)
X = np.array([[0.2, 0.5]])          # Input layer, arbitrary number of neurons
# =============================================================================
# arbitrary number of hidden layers with arbitrary number of neurons within
# =============================================================================
y = np.array([[0.8, 0.1, 0.5, 0.7]])    # Output layer, arbitrary number of neurons


class Neural_Network(object):
    def __init__(self,  layers = [], activationFunction = None):
        '''List layers is a list of integers where every integer represents the number of neurons in the layer.
        The first integer represents the number of neurons in the INPUT layer and the last one number of neurons in the OUTPUT layer.
        Integers in-between the first and the last one represent the number of neurons in hidden layers if there are.
        The second integer in the list represents the neurons of the first hidden layer and penultimate integer the neurons
        in the last hidden layer respectively.
        Default activationFunction is linear. Also, others could be used: activationFunction = "sigmoid" or "ReLU" or "TanH"'''
        self.layers = layers  # list of layers
        self.activationFunction = activationFunction

        # Create a list of weights matrices from the “standard normal” distribution
        self.W = []
        for i in range(len(self.layers)-1):
            self.W.append(np.random.randn(self.layers[i], self.layers[i+1]))
            
        # Biases (random starting parameters)
        self.B = []
        for i in range(len(self.layers)-1):
            self.B.append(np.random.randn(1, self.layers[i+1]))
        
        # plot lists will be appended during traininng procedure
        self.iterations = []
        self.outputValue = []
        self.errorPercent = []
        
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
                    
    def forward(self, X):
        "Propagate inputs through the network"
        self.Z = [np.dot(X,self.W[0])+self.B[0]]  # list of neuron z values (the first one is diferent from template)
        self.A = [self.activation(self.Z[0])]     # list of activated neurons
        for i in range(len(self.layers)-2):
            self.Z.append(np.dot(self.A[i],self.W[i+1])+self.B[i+1])    # apply weights W, add biases B to the neurons
            self.A.append(self.activation(self.Z[i+1]))                 # apply activationFunction
        return self.A[-1]       # predicted output
   
    def costFunction(self, y):
        "Returns cost function value for multiple output"
        J = 0.5*((self.A[-1] - y)**2)
        return J.sum()   # sum of the elements in matrix J      
    
    def costFunctionPrime(self, X, y):
        "Compute derivatives with respect to the weights W and biases B"
        self.delta = [(self.A[-1] - y)*self.activationPrime(self.Z[-1])]   # list of delta values (the last one is diferent from template), * Hadamard product
        self.dJdW = []                          # list of derivatives w.r.t. the weights
        for i in range(1,len(self.layers)-1):   # range from 1 enable insert -1(last) element from a list
            self.delta.insert(0, (np.dot(self.delta[-i], self.W[-i].T))*self.activationPrime(self.Z[-i-1]))  # insert element to the first place in the list self.delta
            self.dJdW.insert(0, np.dot(self.A[-i-1].T, self.delta[-i])) # insert element to the first place in the list self.dJdW
        self.dJdW.insert(0, np.dot(X.T, self.delta[0]))  # the first one is diferent from template
        self.dJdB = self.delta
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
            self.W[i] = np.reshape(weights[boundary[i]:boundary[i+1]],(self.layers[i], self.layers[i+1]))
            # the firt part in () inserts elements inside boundaries
            # the second one defines matrix dim for inserted elements
            
    def getB(self):
        "return list of biases merged into single array"
        # ravel converts matrix to array, np.concatenate creates a single array
        biases = np.concatenate([i.ravel() for i in self.B])  # par -> parameter
        return biases
    
    def setB(self, biases):
        "set new biases from a single biases vector"
        boundary = [0]  # boundaries for cutting matrices from a singe vector
        for i in range(len(self.layers)-1):
            boundary.append(boundary[i] + self.layers[i+1])
            self.B[i] = np.reshape(biases[boundary[i]:boundary[i+1]],(1, self.layers[i+1]))
            # the firt part in () inserts elements inside boundaries
            # the second one defines matrix dim. for inserted elements
                                      
    def gradW(self, X, y):
        "return partial derivatives w.r.t. the weights merged into single array(weight gradient)"
        dJdW = self.costFunctionPrime(X,y)
        return np.concatenate([i.ravel() for i in dJdW])
    
    def gradB(self):
        "return partial derivatives w.r.t. the biases merged into single array(bias gradient)"
        return np.concatenate([i.ravel() for i in self.dJdB])

    def trainNetwork(self, alpha, maxIteration):    # using Gradient Descent (GD)
        '''alpha represents learning rate in range [0,1] and
        maxIteration represents maximal number of iterations'''
        self.alpha = alpha      # learning rate(from 0.0 to 1.0)
        self.maxIteration = maxIteration    # arbitrary 
        currentCost = 1000000       # set any "big" number just to pass first iteration
# =============================================================================
#         currentW = self.getW()
#         currentB = self.getB()
# =============================================================================
        for i in range(1, self.maxIteration + 1): # start range from 1 instead 0, set first iteration as 1 in plot
            self.forward(X)
# =============================================================================
#             if self.costFunction(y) >= currentCost:   # looping until Cost Function start increasing
#                 self.setW(currentW) # set weights from previous iteration where Cost Function was at a minimum
#                 self.setB(currentB) # set biases from previous iteration where Cost Function was at a minimum
#                 print("\nOptimization completed in", len(self.iterations) ,"iterations!\n")                
#                 break
# =============================================================================
            self.iterations.append(i)
            self.outputValue.append(self.forward(X))
            self.errorPercent.append(self.costFunction(y))
            currentCost = self.costFunction(y)
            currentW = self.getW()  # weights before applying GD
            currentB = self.getB()  # biases before applying GD
            newW = currentW - self.alpha * self.gradW(X, y)   # apply GD to get new weights
            newB = currentB -self.alpha * self.gradB()        # apply GD to get new biases
            self.setW(newW)         # set weights for the next iteration
            self.setB(newB)         # set biases fot the next iteration

    def plotOutput(self):
        plt.figure("Output")
        plt.title(self.activationFunction)
        plt.plot(list(range(1, y.size + 1)), self.outputValue[-1][0],'ro', label = 'Predicted')  # 'ro' - red dot
        plt.plot(list(range(1, y.size + 1)), y[0],'bo', label = 'Real')  # 'bo' - blue dot, y.size - number of the elements in matrix y
        plt.xlabel("Output Element")
        plt.ylabel ("Output Value")
        plt.legend()
        plt.grid(True)
        
    def plotError(self):
        plt.figure("Error")
        plt.title(self.activationFunction)
        plt.plot(self.iterations, self.errorPercent)
        plt.xlabel("Iterations")
        plt.ylabel ("Error")
        plt.grid(True)
            
            
print("\n<< Neural Network Training >>\n")                   
NN = Neural_Network([2,4,3,2,4], activationFunction = "ReLU")  # number of neurons in input, hidden and output layer(s)
NN.setW([0.1, 0.55, 0.5, 0.8, 0.1, 0.4, 0.25, 0.35,\
           0.4, 0.2, 0.2, 0.5, 0.1, 0.5, 0.2, 0.1, 0.5, 0.1, 0.2, 0.5,\
           0.1, 0.4, 0.25, 0.4, 0.4, 0.65,\
           0.1, 0.4, 0.5, 0.25, 0.45, 0.25, 0.5, 0.15]) # set specific initial weights
NN.setB([0.4, 0.6, 0.2, 0.15,\
         0.1, 0.3, 0.2,\
         0.2, 0.2,\
         0.4, 0.5, 0.2, 0.3]) # set specific initial biases
startTime = time.process_time()
NN.trainNetwork(0.1, 30)
endTime = time.process_time()
print("Predicted Training Value: \n", NN.forward(X), "\n")
print("Real Training Values: \n", y,"\n")
print("Activation Function: ", NN.activationFunction)
print("Learning Rate: ", NN.alpha)
print("Cost Function: ", NN.costFunction(y))
NN.plotOutput()
NN.plotError()
print("Training time: ", endTime - startTime)

