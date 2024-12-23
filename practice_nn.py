import numpy as np
#single neuron
inputs = [1,2,3,4,5] #shape (5,)
weights = [0.2, 0.8, -0.5, 1.0, 0.7] #shape (5,)
bias = 2

output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] +inputs[3] * weights[3] +inputs[4] * weights[4] + bias
print(output)  

# multiple neurons
inputs = [1,2,3,4,5] #shape (5,)
weights = [[0.2, 0.8, -0.5, 1.0, 0.7], [1.2, -2.1, 3.2, 2.1, 2.2], [1.4, -.21, -.42, 1.2, .7]] #shape (3,5)
biases = [1.2, -2, 3]

output = np.dot(inputs, np.array(weights).T) + biases
print(output)


#batches

inputs = [[1,2,3,4,5], [2.1, 3, -2.1, 2.2, 3], [1.2, 2.1, 3.0, 2.2, 1.2]] #shape (3,5)
weights1 = [[0.2, 0.8, -0.5, 1.0, 0.7], [1.2, -2.1, 3.2, 2.1, 2.2], [1.4, -.21, -.42, 1.2, .7]] #3,5
biases1 = [1.2, -2, 3]
weights2 = [[0.2, 0.8, -0.5, ], [1.2, 2.1, 2.2], [ -.42, 1.2, .7]] #3,5
biases2 = [1.2, -2, 3]
input_layer1= np.dot(inputs, np.array(weights1).T) +biases
print(input_layer1)

input_layer2 = np.dot(input_layer1, np.array(weights2).T) + biases2
print(input_layer2)

## Let's make classes

X = [[1,2,3,4,5], [2.1, 3, -2.1, 2.2, 3], [1.2, 2.1, 3.0, 2.2, 1.2]]

class Layer_dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + bias


layer1 = Layer_dense(5,10)
layer2 = Layer_dense(10,2)

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)
