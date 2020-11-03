# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:13:56 2020

@author: LENOVO
"""

import numpy as np
import pylab as plt


"Example 2: 1 input - signle neuron hidden layers - 1 output"

# y = 0.5*x 

# Training Data
X = 2   # Input layer, 1 neuron
# =============================================================================
# 1st hidden layer (neuron z1)
# 2nd hidden layer (neuron z2)
# 3rd hidden layer (neuron z3)
# =============================================================================
y = 1   # Output layer, 1 neuron
     
    
class Neural_Network(object):
    def __init__(self):
        "Weights (random initial parameters, no biases"
        # Return floats from the “standard normal” distribution
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()
        self.w4 = np.random.randn()
        
        # plot lists
        self.iterations = []
        self.outputValue = []
        self.errorPercent = []

    def forward(self, X):
        "Propagate inputs through the network"
        self.z1 = X * self.w1
        self.z2 = self.z1 * self.w2
        self.z3 = self.z2 * self.w3
        self.yHat = self.z3 * self.w4     
        return self.yHat
   
    def costFunction(self, y):
        "returns Cost Function value"
        return 0.5*(self.yHat - y)**2
           
    def costFunctionPrime(self, X, y):
        "Compute derivatives with respect to w1, w2, w3, w4"
        dJdw4 = (self.yHat - y)*self.z3
        dJdw3 = (self.yHat - y)*self.w4*self.z2
        dJdw2 = (self.yHat - y)*self.w4*self.w3*self.z1
        dJdw1 = (self.yHat - y)*self.w4*self.w3*self.w2*X
        return dJdw1, dJdw2, dJdw3, dJdw4
        
    def getPar(self):
        "returns vector of weights W"
        par = np.array([self.w1, self.w2, self.w3, self.w4])
        return par
    
    def setPar(self, par):
        "set new weights from an array of weights"
        self.w1 = par[0]
        self.w2 = par[1]
        self.w3 = par[2]
        self.w4 = par[3]
        
    def computeGrad(self, X, y):
        "returns array of Cost Function derivatives w.r.t the weights"
        dJdw1, dJdw2, dJdw3, dJdw4 = self.costFunctionPrime(X, y)
        return np.array([dJdw1, dJdw2, dJdw3, dJdw4])
               
    def trainNetwork(self, alpha, maxIteration):    # using Gradient Descent (GD)
        '''alpha represents learning rate in range [0,1] and
        maxIteration represents maximal number of iterations'''
        self.alpha = alpha      # learning rate(from 0.0 to 1.0)
        self.maxIteration = maxIteration    # arbitrary 
        currentCost = 100       # set any "big" number just to pass first iteration
        currentPar = self.getPar()
        for i in range(1, self.maxIteration + 1):
            self.forward(X)
            if self.costFunction(y) >= currentCost:   # looping until Cost Function start increasing
                self.setPar(currentPar) # set weights from previous iteration where Cost Function was at a minimum
                print("\nOptimization completed in", len(self.iterations) ,"iterations!\n")                
                break
            self.iterations.append(i)
            self.outputValue.append(self.forward(X))
            self.errorPercent.append(100 * self.costFunction(y))
            currentCost = self.costFunction(y)
            currentPar = self.getPar()  # weights before applying GD
            newPar = currentPar - self.alpha * self.computeGrad(X, y)   # apply GD to get new weights
            self.setPar(newPar)         # set weights for the new iteration

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
NN.setPar([0.8, 0.6, 0.1, 0.5]) 
NN.trainNetwork(0.1, 20)      # convergence exist for alpha <= 0.32
print("\nPredicted Value: ", NN.forward(X))
print("Real Value :", y,"\n")
print("Learning Rate: ", NN.alpha)
print("Weights: ", NN.getPar())
print("Gradient Magnitude: ", np.linalg.norm(NN.computeGrad(X, y)))
print("Cost Function: ", NN.costFunction(y))
NN.plotOutput()
NN.plotError()
