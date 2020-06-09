from node import Node
import numpy as np
def dot(v1, v2):
    return sum([i*j for (i,j) in zip(v1,v2)])

def relu(value):
    return max(0, value)

inputs, hidden, output = [], [[]], [] 
def create_network(nInputs, nHidden, nOutputs):
    for i in range(nInputs):
        inputs.append(Node(0, np.random.uniform(-1, 1, size=(nHidden[0],)), 0))
    for i in range(len(nHidden)):
        for j in range(nHidden[i]):
            if i + 1 == len(nHidden):
                hidden[i].append(Node(0, np.random.uniform(-1, 1, size=(nOutputs,)), 0.01)) # change to do kaiming initialization
            else:
                hidden[i].append(Node(0, np.random.uniform(-1, 1, size=(nHidden[i+1],)), 0.01))
    for i in range(nOutputs):
        inputs.append(Node(0, [], 0.01))
create_network(2, [3], 2)

# testing
inputs = [Node(0, [0.2, 0.4, 0.2], 0), Node(0, [0.2, 0.4, 0.2], 0)]
hidden = [Node(0, [0.1, 0.4], 0.1), Node(0, [0.2, 0.5], 0.2), Node(0, [0.2, 0.4], 0.1)]
output = [Node(0, [], 0.1), Node(0, [], 0.2)]

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

forward_pass([inputs, hidden, output])
output[0].print_node()