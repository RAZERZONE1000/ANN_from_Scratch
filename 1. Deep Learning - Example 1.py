# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:13:56 2020

@author: LENOVO
"""

import numpy as np
import pylab as plt


"Example 1: 1 input - 1 output"

# y = 0.5*x 

# Training Data
X = 2   # Input layer, 1 neuron
y = 1   # Output layer, 1 neuron
    
 
    
class Neural_Network(object):
    def __init__(self):     
        "Signle weight (random starting parameter), no bias"
        self.w = np.random.randn()  # Return float from the “standard normal” distribution
        
        # plot lists
        self.iterations = []
        self.outputValue = []
        self.errorPercent = []

    def forward(self, X):
        "Propagate inputs through the network"
        self.yHat = X * self.w      # apply weight to the input layer neuron
        return self.yHat
   
    def costFunction(self, X, y):
        "returns cost function value"
        return 0.5*(y - self.yHat)**2
    
    def costFunctionPrime(self, X, y):
        "Compute derivatives with respect to w"
        dJdw = (self.yHat - y)*X    # dJdw = dJdyHat * dyHatdw
        return dJdw
    
    def getPar(self):
        "returns weight w"
        return self.w
    
    def setPar(self, par):
        "set new weight w"
        self.w = par
               
    def trainNetwork(self, alpha, maxIteration):    # using Gradient Descent
        '''alpha represents learning rate in range [0,1] and
        maxIteration represents maximal number of iterations'''
        self.alpha = alpha      # learning rate(from 0.0 to 1.0)
        self.maxIteration = maxIteration    # arbitrary 
        currentCost = 100       # set any "big" number just to pass first iteration
        for i in range(self.maxIteration):
            self.iterations.append(i)
            self.forward(X)
            self.outputValue.append(self.forward(X))
            self.errorPercent.append(100 * self.costFunction(X, y))
            currentPar = self.getPar()
            newPar = currentPar - self.alpha * self.costFunctionPrime(X, y)
            self.setPar(newPar)
            if self.costFunction(X,y) >= currentCost:   # looping until Cost FUnction start increasing
                print("\nOptimization completed in", len(self.iterations) ,"iterations!\n")
                break              
            currentCost = self.costFunction(X,y)
       
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
                    
NN = Neural_Network()
NN.setPar(0.95)
NN.trainNetwork(0.1, 20)      # convergence exist for alpha < 0.5
print("\nPredicted Value: ", NN.forward(X), "\n")
print("Real Value :", y,"\n")
print("Learning Rate: ", NN.alpha)
print("Weight: ", NN.getPar())
print("Cost Function Prime: ", NN.costFunctionPrime(X, y))
print("Cost Function: ", NN.costFunction(X, y))
NN.plotOutput()
NN.plotError()



