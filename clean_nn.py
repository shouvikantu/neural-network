import numpy as np
import nnfs
from nnfs.datasets import spiral_data


#X = [[1,2,3,2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(samples=100, classes=3)




class Dense_Layer:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights= 0.10* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax():
    def forward(self, inputs):
        exp_values =  np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output= probabilities


dense1= Dense_Layer(2, 3)
activation1 = Activation_ReLu()

dense2 = Dense_Layer(3,3)
activation2 = Activation_Softmax()


dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])