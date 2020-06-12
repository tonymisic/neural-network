from node import Node
import numpy as np
from math import sqrt, exp, log
# globals
inputs, hidden, output = [], [], []
# hyper-parameters
intial_bias = 0.1
learning_rate = 0.2
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

# populates network with empty nodes
def create_network(nInputs, nHidden, nOutputs):
    for i in range(nInputs):
        inputs.append(Node(0, np.random.uniform(-1, 1, size=(nHidden[0],))*sqrt(2./nInputs), 0)) # Kaiming initialization of weights
    for i in range(len(nHidden)):
        hidden.append([])
        for _ in range(nHidden[i]):
            if i + 1 == len(nHidden):
                hidden[i].append(Node(0, np.random.uniform(-1, 1, size=(nOutputs,))*sqrt(2./nHidden[i]), intial_bias)) 
            else:
                hidden[i].append(Node(0, np.random.uniform(-1, 1, size=(nHidden[i+1],))*sqrt(2./nHidden[i]), intial_bias))
    for i in range(nOutputs):
        output.append(Node(0, [], intial_bias))
    print("Network Created ... Architecture: Input - " + str(nInputs) + " Hidden - " + str(nHidden) + " Output - " + str(nOutputs))

def set_input(current):
    if len(current) == len(inputs):
        for i in range(len(inputs)):
            inputs[i].value = current[i]
    else:
        raise Exception("Input not set, incorrect input size of: " + str(len(current)))

# performs one forward pass of the network
def forward_pass(X):
    layers = []
    set_input(X)
    layers.append(inputs)
    for i in hidden:
        layers.append(i)
    layers.append(output)
    for i in range(len(layers) - 1): # each layer in the network
        node_count = 0
        for j in layers[i + 1]: # get next layer's nodes and set their values
            new_value = 0
            for k in layers[i]: # get current layers nodes and calculate value of next layer's node
                new_value += k.value * k.weights[node_count]
            j.value = sigmoid(new_value + j.bias) # add bias and relu
            node_count += 1
    return output[0].value

def back_propagate(Y):
    # calculate loss
    costs = []
    costs_derivatives = []
    if len(Y) == len(output):
        for n in range(len(output)):
            costs.append(error(Y[n], output[n].value))
            costs_derivatives.append(error_derivative(Y[n], output[n].value))
    else:
        raise Exception("Output not set, incorrect output size of: " + str(len(Y)))
    layers = []
    layers.append(inputs)
    for i in hidden:
        layers.append(i)
    layers.append(output)
    # # neuron based changes hidden to output
    # delta_h_b = sigmoid_derivative(dot([hidden[0][0].weights[0]], [hidden[0][0].value]) + output[0].bias) * costs_derivatives[0]
    # delta_h1_w = hidden[0][0].value * sigmoid_derivative(dot([hidden[0][0].weights[0]], [hidden[0][0].value]) + output[0].bias) * costs_derivatives[0] 
    # delta_h2_w = hidden[0][1].value * sigmoid_derivative(dot([hidden[0][1].weights[0]], [hidden[0][1].value]) + output[0].bias) * costs_derivatives[0]
    # calculate deltas for each weight and bias
    for layer in range(len(layers) - 1, 1, -1): # starting at output layer go back
        layer_deltas = []
        if layer == len(layers) - 1: # output layer
            node_count = 0
            for j in layers[layer]:
                new_value = 0
                for k in layers[layer - 1]:
                    new_value += k.value * k.weights[node_count]
                delta = sigmoid_derivative(new_value + j.bias) * costs_derivatives[0]
                node_count += 1
        else: # All layers but the ouput
            node_count = 0
            for j in layers[layer]:
                new_value = 0
                for k in layers[layer - 1]:
                    new_value += k.value * k.weights[node_count]
                delta = sigmoid_derivative(new_value + j.bias) 
                node_count += 1

def train(X, Y, epochs):
    for e in range(epochs):
        s = 0
        for i in range(len(X)):
            forward_pass(X[i])
            back_propagate(Y[i])
        print("Epoch " + str(e + 1))

def test(sample, expected):
    pass
# create empty slate
create_network(2, [2], 1)
# Intialize XOR gate datasets
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
# train on XOR
train(X,Y, 200)
print(forward_pass([0, 0]))
print(forward_pass([0, 1]))
print(forward_pass([1, 0]))
print(forward_pass([1, 1]))
