from node import Node
import numpy as np
from math import sqrt
# globals
inputs, hidden, output = [], [], [] 
# hyper-parameters
intial_bias = 0.01
learning_rate = 0.1

# dot product
def dot(v1, v2):
    return sum([i*j for (i,j) in zip(v1,v2)])

# rectified linear unit (ReLU)
def relu(value):
    return max(0, value)

# populates network with empty nodes
def create_network(nInputs, nHidden, nOutputs):
    for i in range(nInputs):
        inputs.append(Node(0, np.random.uniform(-1, 1, size=(nHidden[0],))*sqrt(2./nInputs), 0)) # Kaiming initialization of weights
    for i in range(len(nHidden)):
        hidden.append([])
        for j in range(nHidden[i]):
            if i + 1 == len(nHidden):
                hidden[i].append(Node(0, np.random.uniform(-1, 1, size=(nOutputs,))*sqrt(2./nHidden[i]), intial_bias)) 
            else:
                hidden[i].append(Node(0, np.random.uniform(-1, 1, size=(nHidden[i+1],))*sqrt(2./nHidden[i]), intial_bias))
    for i in range(nOutputs):
        output.append(Node(0, [], intial_bias))
    print("Network Created ... Architecture: Input - " + str(nInputs) + " Hidden - " + str(nHidden) + " Output - " + str(nOutputs))

# performs one forward pass of the network
def forward_pass(layers):
    for i in range(len(layers) - 1): # each layer in the network
        node_count = 0
        for j in layers[i + 1]: # get next layer's nodes and set their values
            new_value = 0
            for k in layers[i]: # get current layers nodes and calculate value of next layer's node
                new_value += k.value * k.weights[node_count]
            j.value = relu(new_value + j.bias) # add bias and relu
            node_count += 1
    print("Forward Pass Complete!")

def back_propagate(layers, truths):
    pass

def train(X, Y, model, epochs):
    pass

# create empty slate
create_network(2, [2], 1)
# one forward pass of training
forward_pass([inputs, hidden[0], output])
output[0].print_node()
