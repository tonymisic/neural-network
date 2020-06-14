from node import Node
import numpy as np
from math import sqrt, exp, log, floor, ceil
# globals
weights, gradient_weights = [], []
biases, gradient_biases = [], []
activations = []
# hyper-parameters
intial_bias = 0.01
learning_rate = 0.01
# dot product
def dot(v1, v2):
    return sum([i*j for (i,j) in zip(v1,v2)])
# rectified linear unit (ReLU)
def relu(value):
    return max(0, value)
def relu_derivative(value):
    if value <= 0:
        return 0
    else:
        return 1
# sigmoid activation function
def sigmoid(value):
    return exp(value) / (exp(value) + 1)
def sigmoid_derivative(value):
    return sigmoid(value) * (1 - sigmoid(value))
# softmax, returns normalized list of floats
def softmax(L):
    result = []
    exps = []
    for a in L:
        exps.append(exp(a))
    for a in exps:
        result.append(a / sum(exps))
    return result
# Root Mean Square Error (RMSE)
def rmse(Y):
    temp = []
    if len(Y) == len(activations[-1]):
        for n in range(len(activations[-1])):
            temp.append(Y[n] * log(activations[-1][n].value))
    else:
        raise Exception("Output not set, incorrect output size of: " + str(len(Y)))
    return -sum(temp)
# Square Error function
def error(truth, predicted):
    return 0.5 * (predicted - truth) ** 2
def error_derivative(truth, predicted):
    return predicted - truth

# populates network with empty values using random initialization of weights
def create_network(nInputs, nHidden, nOutputs):
    weights.append(np.random.uniform(-1, 1, size=(nInputs * nHidden[0],)))
    gradient_weights.append(np.random.uniform(0, 0, size=(nInputs * nHidden[0],)))
    biases.append(np.random.uniform(intial_bias, intial_bias, size=(nHidden[0],)))
    gradient_biases.append(np.random.uniform(0, 0, size=(nHidden[0],)))
    activations.append(np.random.uniform(0, 0, size=(nInputs)))
    for i in range(len(nHidden)):
        if i + 1 == len(nHidden):
            weights.append(np.random.uniform(-1, 1, size=(nHidden[i] * nOutputs,)))
            gradient_weights.append(np.random.uniform(0, 0, size=(nHidden[i] * nOutputs,)))
            activations.append(np.random.uniform(0, 0, size=(nHidden[i])))
        else:
            weights.append(np.random.uniform(-1, 1, size=(nHidden[i] * nHidden[i + 1],)))
            gradient_weights.append(np.random.uniform(0, 0, size=(nHidden[i] * nHidden[i + 1],)))
            biases.append(np.random.uniform(intial_bias, intial_bias, size=(nHidden[i + 1],)))
            gradient_biases.append(np.random.uniform(0, 0, size=(nHidden[i + 1],)))
            activations.append(np.random.uniform(0, 0, size=(nHidden[i],)))
    biases.append(np.random.uniform(intial_bias, intial_bias, size=(nOutputs,)))
    gradient_biases.append(np.random.uniform(0, 0, size=(nOutputs,)))
    activations.append(np.random.uniform(0, 0, size=(nOutputs,)))
    print("Network Created ... Architecture: Input - " + str(nInputs) + " Hidden - " + str(nHidden) + " Output - " + str(nOutputs))

def set_input(current):
    activations[0] = current

# performs one forward pass of the network
def forward_pass(X):
    set_input(X)
    for i in range(len(biases)):
        for j in range(len(biases[i])):
            L = len(biases[i])
            activations[i + 1][j] = sigmoid(dot(weights[i][j*L:j*L+L], activations[i]) + biases[i][j])
    return activations[-1][0]

def back_propagate(Y):
    loss_der = error_derivative(Y[0], activations[-1][0])
    # calculate gradients
    for i in reversed(range(len(weights))):
        if i == len(weights) - 1: # output layer
            for j in range(len(biases[i])):
                L = len(biases[i])
                gradient_biases[i][j] = sigmoid_derivative(dot(weights[i][j*L:j*L+L], activations[i]) + biases[i][j]) * loss_der
                for k in range(len(weights[i])):
                    gradient_weights[i][k] = activations[i][j] * sigmoid_derivative(dot(weights[i][k*L:k*L+L], activations[i]) + biases[i][j]) * loss_der
        else: # all other layers
            for j in range(len(biases[i])): # 2 times
                L = len(biases[i])
                gradient_biases[i][j] = sigmoid_derivative(dot(weights[i][j*L:j*L+L], activations[i]) + biases[i][j]) * loss_der
                for k in range(len(weights[i])): # 4 times
                    gradient_weights[i][k] = activations[i][j] * sigmoid_derivative(dot(weights[i][k*L:k*L+L], activations[i]) + biases[i][j]) * loss_der
    # update biases with learning rate
    for i in range(len(gradient_biases)):
        for j in range(len(gradient_biases[i])):
            biases[i][j] += learning_rate * gradient_biases[i][j]
    # update weights with learning rate
    for i in range(len(gradient_weights)):
        for j in range(len(gradient_weights[i])):
            weights[i][j] += learning_rate * weights[i][j]
    return loss_der

def train(X, Y, epochs):
    for e in range(epochs):
        s = 0
        for i in range(len(X)):
            forward_pass(X[i])
            s = back_propagate(Y[i])
        print("Epoch " + str(e + 1))
        
def print_network():
    print(" - - - Network - - - ")
    print("Weights: " + str(weights))
    print("Gradients: " + str(gradient_weights))
    print("Biases" + str(biases))
    print("Gradients: " + str(gradient_biases))
    print("Activations: " + str(activations))
    print(" - - - Network - - - ")
def test(sample, expected):
    pass

# create empty slate
create_network(2, [2], 1)
# Intialize XOR gate datasets
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
# train on XOR
train(X,Y, 100)
print_network()
# show result
print("XOR Table Results: ")
print('{:f}'.format(forward_pass([0,0])))
print('{:f}'.format(forward_pass([0,1])))
print('{:f}'.format(forward_pass([1,0])))
print('{:f}'.format(forward_pass([1,1])))