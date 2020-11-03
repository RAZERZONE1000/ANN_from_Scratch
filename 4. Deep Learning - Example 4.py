# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:13:56 2020

@author: LENOVO
"""

# Import libraries
import numpy as np
import pylab as plt


"Example 4: multiple inputs - multiple neurons hidden layers - one output"

# Training Data (1 sample)
X = np.array([[2, 4]])   # Input layer, arbitrary number of neurons
# =============================================================================
# arbitrary number of hidden layers with arbitrary number of neurons within
# =============================================================================
y = np.array([[8]])      # Output layer, 1 neuron


class Neural_Network(object):
    def __init__(self,  layers = []):
        '''List layers is a list of integers where every integer represents the number of neurons in the layer.
        The first integer represents the number of neurons in the INPUT layer and the last one number of neurons in the OUTPUT layer.
        Integers in-between the first and the last one represent the number of neurons in hidden layers if there are.
        The second integer in the list represents the neurons of the first hidden layer and penultimate integer the neurons
        in the last hidden layer respectively.'''
        self.layers = layers  # list of layers

        # Create list of weights matrices from the “standard normal” distribution
        self.W = []
        for i in range(len(self.layers)-1):
            self.W.append(np.random.randn(self.layers[i], self.layers[i+1]))
        
        # plot lists will be appended during traininng procedure
        self.iterations = []
        self.outputValue = []
        self.errorPercent = []

        
    def forward(self, X):
        "Propagate inputs through the network"
        self.Z = [np.dot(X,self.W[0])]  # list of neuron values (the first one is diferent from template)
        for i in range(len(self.layers)-2):
            self.Z.append(np.dot(self.Z[i],self.W[i+1])) # apply weights W to the neurons
        return self.Z[-1]       # predicted output
   
    def costFunction(self, y):
        "return cost function value for one input sample"
        return 0.5*(self.Z[-1] - y)**2
    
    def costFunctionPrime(self, X, y):
        "Compute derivatives with respect to the weights W"
        self.delta = [self.Z[-1] - y]   # list of delta values (the last one is diferent from template)
        self.dJdW = []                  # list of derivatives w.r.t. the weights
        for i in range(1,len(self.layers)-1):
            self.delta.insert(0, np.dot(self.delta[-i], self.W[-i].T))  # insert element to the first place in the list self.delta
            self.dJdW.insert(0, np.dot(self.Z[-i-1].T, self.delta[-i])) # insert element to the first place in the list self.dJdW
        self.dJdW.insert(0, np.dot(X.T, self.delta[0]))  # the first one is diferent from template
        return self.dJdW
    
    def getPar(self):
        "return list of weights merged into single array"
        # ravel converts matrix to array, np.concatenate creates a single array
        par = np.concatenate([i.ravel() for i in self.W])  # par -> parameter
        return par
    
    def setPar(self, par):
        "set new weights from a single parameter vector"
        boundary = [0]  # boundaries for cutting matrices from a singe vector
        for i in range(len(self.layers)-1):
            boundary.append(boundary[i] + self.layers[i]*self.layers[i+1])
            self.W[i] = np.reshape(par[boundary[i]:boundary[i+1]],(self.layers[i], self.layers[i+1]))
            # the firt part in () inserts elements inside boundaries
            # the second one defines matrix dim for inserted elements
    
    def computeGrad(self, X, y):
        "return partial derivatives merged into single array(gradient)"
        dJdW = self.costFunctionPrime(X,y)
        return np.concatenate([i.ravel() for i in dJdW])

    def trainNetwork(self, alpha, maxIteration):    # using Gradient Descent (GD)
        '''alpha represents learning rate in range [0,1] and
        maxIteration represents maximal number of iterations'''
        self.alpha = alpha      # learning rate(from 0.0 to 1.0)
        self.maxIteration = maxIteration    # arbitrary 
        currentCost = 1000000       # set any "big" number just to pass first iteration
        currentPar = self.getPar()
        for i in range(1, self.maxIteration + 1): # start range from 1 instead 0, set first iteration as 1 in plot
            self.forward(X)
# =============================================================================
#             if self.costFunction(y) >= currentCost:   # looping until Cost Function start increasing
#                 self.setPar(currentPar) # set weights from previous iteration where Cost Function was at a minimum
#                 print("\nOptimization completed in", len(self.iterations) ,"iterations!\n")                
#                 break
# =============================================================================
            self.iterations.append(i)
            self.outputValue.append(float(self.forward(X)))
            self.errorPercent.append(float(100 * self.costFunction(y)))
            currentCost = self.costFunction(y)
            currentPar = self.getPar()  # weights before applying GD
            newPar = currentPar - self.alpha * self.computeGrad(X, y)   # apply GD to get new weights
            self.setPar(newPar)         # set weights for the next iteration

    def plotOutput(self):
        plt.figure("Output")
        plt.plot(self.iterations, self.outputValue)
        plt.xlabel("Iterations")
        plt.ylabel ("Output Value")
        plt.grid(True)
        
    def plotError(self):
        plt.figure("Error")
        plt.plot(self.iterations, self.errorPercent)
        plt.xlabel("Iterations")
        plt.ylabel ("Error (%)")
        plt.grid(True)
            
            
print("\n<< Neural Network Training >>\n")                   
NN = Neural_Network([2,2,2,1])  # number of neurons in input, hidden and output layer(s)
NN.setPar([0.7, 0.5, 0.1, 0.9, 0.4, 0.6, 0.5, 0.2, 0.8, 0.4]) # set specific initial weights
NN.trainNetwork(0.005, 15)  
print("Predicted Training Value: \n", float(NN.forward(X)), "\n")
print("Real Training Values: \n", int(y),"\n")
print("Learning Rate: ", NN.alpha)
print("Gradient Magnitude: ", np.linalg.norm(NN.computeGrad(X, y)))
print("Cost Function: ", float(NN.costFunction(y)))
NN.plotOutput()
NN.plotError()

