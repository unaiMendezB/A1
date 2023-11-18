import numpy as np
import pandas as pd

# Neural Network class
class MyNeuralNetwork:
  def __init__(self, layers, epochs, learning_rate, momentum, activation_function, validation_set_percentage):

    self.L = len(layers)    # Number of layers
    self.n = layers   # Number of units in each layer
    self.epochs = epochs    # Number of epochs for training
    self.learning_rate = learning_rate    # Learning rate for the network
    self.momentum = momentum    # Momentum for the weight updates
    self.fact = activation_function        # Activation function to be used in the network (sigmoid, relu, linear, tanh)
    self.validation_set_percentage = validation_set_percentage        # Percentage of data to be used as validation set

    self.xi = []  # Node values

    self.w = []    # Weights for each layer
    self.w.append(np.zeros((1, 1)))

    self.theta = []  # Thresholds for each layer
    self.theta.append(np.zeros((1, 1)))

    self.delta = []  # Error propagation for each layer
    self.delta.append(np.zeros((1, 1)))

    self.d_w = []  # Weight changes for each layer
    self.d_w.append(np.zeros((1, 1)))

    self.d_theta = []  # Threshold changes for each layer
    self.d_theta.append(np.zeros((1, 1)))

    self.d_w_prev = []  # Previous weight changes for momentum calculation
    self.d_w_prev.append(np.zeros((1, 1)))

    self.d_theta_prev = []  # Previous threshold changes for momentum calculation
    self.d_theta_prev.append(np.zeros((1, 1)))

    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))

    for lay in range(1, self.L):
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))
      self.theta.append(np.zeros(layers[lay]))
      self.delta.append(np.zeros(layers[lay]))
      self.d_w.append(np.zeros((layers[lay], layers[lay - 1])))
      self.d_theta.append(np.zeros(layers[lay]))
      self.d_w_prev.append(np.zeros((layers[lay], layers[lay - 1])))

      self.d_theta_prev.append(np.zeros(layers[lay]))

  # This method allows us to train the network with this data.
  # X an array of size (n_samples,n_features) which holds the training samples represented as floating point feature vectors
  # y of size (n_samples), which holds the target values (class labels) for the training samples.
  def fit(self, X, y):

    return

  # X an array of size (n_samples,n_features) that contains the samples.
  # This method returns a vector with the predicted values for all the input samples
  def predict(self, X):

    return

  # that returns 2 arrays of size (n_epochs, 2) that contain the evolution of the training error and the validation
  # error for each of the epochs of the system, so this information can be plotted.
  def loss_epochs(self):

    return

'''
1 Scale input and/or output patterns, if needed
2 Initialize all weights and thresholds randomly
3 For epoch = 1 To num epochs
4   For pat = 1 To num training patterns
5     Choose a random pattern (x, z) of the training set
6     Feed􀀀forward propagation of pattern x to obtain the output o(x)
7     Back􀀀propagate the error for this pattern
8     Update the weights and thresholds
9   End For
10  Feed􀀀forward all training patterns and calculate their prediction quadratic error
11  Feed􀀀forward all validation patterns and calculate their prediction quadratic error
12 End For
13 # Optional: Plot the evolution of the training and validation errors
14 Feed􀀀forward all test patterns
15 Descale the predictions of test patterns, and evaluate them
'''

# 1 Scale input and/or output patterns, if needed
dataSynth = pd.read_csv('synthetic_Normalized.txt', sep='\t')
dataTurb = pd.read_csv('turbine_Standardized.txt', sep='\t')

# 2 Initialize all weights and thresholds randomly
# layers include input layer + hidden layers + output layer
layers = [4, 9, 5, 1]
nn = MyNeuralNetwork(layers)











print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")