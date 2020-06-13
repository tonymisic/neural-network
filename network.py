from node import Node
import numpy as np
from math import sqrt, exp, log
# globals
weight_matrix = []
bias_matrix = []
activation_matrix = []
# hyper-parameters
intial_bias = 0.01
learning_rate = 0.1
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
    if len(Y) == len(output):
        for n in range(len(output)):
            temp.append(Y[n] * log(output[n].value))
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
    weight_matrix.append(np.random.uniform(-1, 1, size=(nInputs * nHidden[0],)))
    bias_matrix.append(np.random.uniform(intial_bias, intial_bias, size=(nHidden[0],)))
    activation_matrix.append(np.random.uniform(0, 0, size=(nInputs)))
    for i in range(len(nHidden)):
        if i + 1 == len(nHidden):
            weight_matrix.append(np.random.uniform(-1, 1, size=(nHidden[i] * nOutputs,)))
            activation_matrix.append(np.random.uniform(0, 0, size=(nHidden[i])))
        else:
            weight_matrix.append(np.random.uniform(-1, 1, size=(nHidden[i] * nHidden[i + 1],)))
            bias_matrix.append(np.random.uniform(intial_bias, intial_bias, size=(nHidden[i + 1],)))
            activation_matrix.append(np.random.uniform(0, 0, size=(nHidden[i],)))
    bias_matrix.append(np.random.uniform(intial_bias, intial_bias, size=(nOutputs,)))
    activation_matrix.append(np.random.uniform(0, 0, size=(nOutputs,)))
    print("Network Created ... Architecture: Input - " + str(nInputs) + " Hidden - " + str(nHidden) + " Output - " + str(nOutputs))

def set_input(current):
    activation_matrix[0] = current

# performs one forward pass of the network
def forward_pass(X):
    set_input(X)
    for i in range(len(activation_matrix) - 1):
        for j in range(len(activation_matrix[i + 1])):
            l = len(activation_matrix[i])
            activation_matrix[i + 1][j] = dot(weight_matrix[i][], activation_matrix[i]) + bias_matrix[i][j]
    return activation_matrix[-1][0]

def back_propagate(Y):
    loss = error_derivative(Y[0], activation_matrix[-1][0])

    return loss

def train(X, Y, epochs):
    for e in range(epochs):
        s = 0
        for i in range(len(X)):
            forward_pass(X[i])
            s += back_propagate(Y[i])
        print("Epoch " + str(e + 1) + " Average Error: " + str(s / len(Y)))

def print_network():
    print(" - - - Network - - - ")
    print("Weights: " + str(weight_matrix))
    print("Activations: " + str(activation_matrix))
    print("Biases" + str(bias_matrix))

def test(sample, expected):
    pass
# create empty slate
create_network(2, [2], 1)
# Intialize XOR gate datasets
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
# train on XOR
train(X,Y, 1)
print_network()
# show result
print("XOR Table Results: ")
print(forward_pass([0,0]))
print(forward_pass([0,1]))
print(forward_pass([1,0]))
print(forward_pass([1,1]))