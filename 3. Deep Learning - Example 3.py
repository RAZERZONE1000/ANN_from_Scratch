# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:13:56 2020

@author: LENOVO
"""

# Import libraries
import numpy as np
import pylab as plt


"Example 3: 1 input - multiple neurons hidden layers - 1 output"

# y = 0.5*x  function to be trained

# Training Data
X = np.array([[2]])   # Input layer, arbitrary number of neurons
# =============================================================================
# 1st hidden layer, hi_1 = arbitrary number of neurons
# 2nd hidden layer, hi_2 = arbitrary nubmer of neurons
# =============================================================================
y = 1               # Output layer, 1 neuron


class Neural_Network(object):
    def __init__(self,  inputLayerSize = 1, hi_1 = 1, hi_2 = 1, outputLayerSize = 1):  # 1,1,1,1 default number of neurons by layers
        "Define architecture of the neural network and hyperparameters"
        self.inputLayerSize = inputLayerSize
        self.hidden_1 = hi_1
        self.hidden_2 = hi_2
        self.outputLayerSize = outputLayerSize
        
        # Weights (random starting parameters), no biases
        # Return parameters from the “standard normal” distribution
        self.W1 = np.random.randn(self.inputLayerSize, self.hidden_1) 
        self.W2 = np.random.randn(self.hidden_1, self.hidden_2)
        self.W3 = np.random.randn(self.hidden_2, self.outputLayerSize)
        
        # plot lists
        self.iterations = []
        self.outputValue = []
        self.errorPercent = []
        
    def forward(self, X):
        "Propagate inputs through network"
        self.z2 = np.dot(X, self.W1)           # apply weights to input layer neurons
        self.z3 = np.dot(self.z2, self.W2)     # apply weights to hidden_1 neurons
        self.yHat = np.dot(self.z3, self.W3)   # apply weights to hidden_2 neurons
        return self.yHat                       # predicted output
   
    def costFunction(self, y):
        "Returns cost function value for one input sample"
        return 0.5*(self.yHat - y)**2
    
    def costFunctionPrime(self, X, y):
        "Compute derivatives with respect to W1, W2, W3"
        delta3 = self.yHat-y
        dJdW3 = np.dot(self.z3.T, delta3)
        delta2 = np.dot(delta3, self.W3.T)
        dJdW2 = np.dot(self.z2.T, delta2)
        delta1 = np.dot(delta2, self.W2.T)
        dJdW1 = np.dot(X.T, delta1)
        return dJdW1, dJdW2, dJdW3 
    
    def getPar(self):
        "eturns W1, W2, W3 merged into single array"
        # ravel converts matrix to array, np.concatenate creates a single array
        par = np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel()))  # par -> parameter
        return par
    
    def setPar(self, par):
        "set W1 and W2 from single parameter vector"
        W1_start = 0
        W1_end = self.inputLayerSize * self.hidden_1
        self.W1 = np.reshape(par[W1_start:W1_end],(self.inputLayerSize, self.hidden_1))
        W2_end = W1_end + self.hidden_1 * self.hidden_2
        self.W2 = np.reshape(par[W1_end:W2_end],(self.hidden_1, self.hidden_2))
        W3_end = W2_end + self.hidden_2 * self.outputLayerSize
        self.W3 = np.reshape(par[W2_end:W3_end],(self.hidden_2, self.outputLayerSize))
    
    def computeGrad(self, X, y):
        "returns partial derivatives mergeg into single array(gradient)"
        dJdW1, dJdW2, dJdW3 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel(), dJdW3.ravel()))

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
NN = Neural_Network(1,2,2,1)  # number of neurons in input, hidden and output layer(s)
NN.setPar([0.7, 0.5, 0.4, 0.6, 0.5, 0.2, 0.1, 0.1]) # set specific initial weights
NN.trainNetwork(0.1, 15)  
print("Predicted Training Value: \n", float(NN.forward(X)), "\n")
print("Real Training Values: \n", y,"\n")
print("Learning Rate: ", NN.alpha)
print("Gradient Magnitude: ", np.linalg.norm(NN.computeGrad(X, y)))
print("Cost Function: ", float(NN.costFunction(y)))
NN.plotOutput()
NN.plotError()