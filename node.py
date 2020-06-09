class Node:
    value, weights, bias= 0, [], 0
    def __init__(self, value, weights, bias):
        self.weights = weights
        self.value = value
        self.bias = bias

    def print_node(self):
        print("Value: " + str(self.value))
        print("Weights: " + str(self.weights))
        print("Bias: " + str(self.bias))