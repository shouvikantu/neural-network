# inputs = [1,2,3, 2.5]
# weights1 = [0.2, 0.8, -0.5, 1.0]
# weights2 = [0.5, -0.91, 0.26, -0.5]
# weights3 = [-0.26, -0.27, 0.17, 0.87]

# bias1 = 2
# bias2 = 3
# bias3 = 0.5

# output = [inputs[0]* weights1[0] + inputs[1]* weights1[1] +inputs[2]* weights1[2] +inputs[3]* weights1[3] +bias1, 
#           inputs[0]* weights2[0] + inputs[1]* weights2[1] +inputs[2]* weights2[2] +inputs[3]* weights2[3] +bias2, 
#           inputs[0]* weights3[0] + inputs[1]* weights3[1] +inputs[2]* weights3[2] +inputs[3]* weights3[3] +bias3]
 
#print(output)

## Transferring to np and new code, same problem
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# # For singular neuron

# inputs = [1,2,3,2.5]
# weights = [0.2, 0.8, -0.5, 1.0]
# bias = 2

# output = np.dot(weights , inputs) +bias
# #print(output)


# # For all layers

# inputs = [1,2,3, 2.5]
# weights=[[0.2, 0.8, -0.5, 1.0],[0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]]

# biases = [2,3,0.5]

# output = np.dot(weights, inputs) +biases
# #print(output)

# ## batches of input 
# # Batches: The number of inputs you show the model at one time. 
# inputs = [[1,2,3, 2.5], [2.0, 5.0, -1.0, 2.0],[-1.5, 2.7, 3.3, -0.8]]
# weights=[[0.2, 0.8, -0.5, 1.0],[0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]]

# biases = [2,3,0.5]

# weights2=[[0.1,-0.14,0.5],[-0.5,0.12,-0.33],[-0.44,0.73,-0.13]]

# biases2 = [-1,2,-0.5]

# layer1_output = np.dot(inputs, np.array(weights).T) +biases
# #print(output)

# layer2_output= np.dot(layer1_output, np.array(weights2).T) +biases2
#(layer2_output)

## Objects


#X = [[1,2,3, 2.5], [2.0, 5.0, -1.0, 2.0],[-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100,3)

print(X)

class Layer_Dense:
    def __init__(self, n_inputs:int, n_neurons:int) -> None:
        #Transposing so that we don't need to do an operation
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) 
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+ self.biases
        
class Activation_Relu:
    def forward(self, inputs):
        self.output= np.maximum(0, inputs)

class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1/ (1+ np.exp(-inputs))

layer1 = Layer_Dense(2,5)
activation1 = Activation_Relu()
sigmoid1 = Activation_Sigmoid()
# layer2 = Layer_Dense(5, 3)
# layer3 = Layer_Dense(3,7)

layer1.forward(X)
print(layer1.output)

activation1.forward(layer1.output)
print(activation1.output)
#print(layer1.output)

sigmoid1.forward(layer1.output)
print(sigmoid1.output)
# layer2.forward(layer1.output)
# #print(layer2.output)

# layer3.forward(layer2.output)
# print(layer3.output)


## rectified Linear 

# inputs = [0,2, -1, 3.3, -2.7, 2.2]

# output=[]

# for i in inputs:
#     if i>0:
#         output.append(i)
#     else:
#         output.append(0)
# print(output)

