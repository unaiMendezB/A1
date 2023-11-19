import numpy as np
import pandas as pd


# Neural Network class
class MyNeuralNetwork:
    def __init__(self, layers, epochs, learning_rate, momentum, activation_function, validation_set_percentage):

        self.L = len(layers)  # Number of layers
        self.n = layers  # Number of units in each layer
        self.epochs = epochs  # Number of epochs for training
        self.learning_rate = learning_rate  # Learning rate for the network
        self.momentum = momentum  # Momentum for the weight updates
        self.fact = activation_function  # Activation function to be used in the network (sigmoid, relu, linear, tanh)
        self.validation_set_percentage = validation_set_percentage  # Percentage of data to be used as validation set

        self.xi = []  # Node values

        self.z = []

        self.w = []  # Weights for each layer
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

        for epoch in range(self.epochs):    #  (3)
            for pat in range(X.shape[0]):   #  (4)
                # Choose a random pattern (5)
                x = X[pat]
                z = y[pat]

                # Feed-forward propagation (6)
                self.xi[0] = x
                for lay in range(1, self.L):
                    self.z[lay] = np.dot(self.w[lay], self.xi[lay - 1]) + self.theta[lay]
                    self.xi[lay] = activate(self.fact, self.z[lay])

                # Back-propagate the error (7)
                self.delta[-1] = (self.xi[-1] - z) * activate_derivative(self.fact, self.xi[-1])
                for lay in range(self.L - 2, 0, -1):
                    self.delta[lay] = np.dot(self.w[lay + 1].T, self.delta[lay + 1]) * activate_derivative(self.fact,
                                                                                                           self.xi[lay])

                # Update the weights and thresholds (8)
                for lay in range(1, self.L):
                    self.d_w[lay] = np.dot(self.delta[lay], self.xi[lay - 1].T)
                    self.d_theta[lay] = self.delta[lay]
                    self.w[lay] -= self.learning_rate * self.d_w[lay] + self.momentum * self.d_w_prev[lay]
                    self.theta[lay] -= self.learning_rate * self.d_theta[lay] + self.momentum * self.d_theta_prev[lay]

                    self.d_w_prev[lay] = self.d_w[lay]
                    self.d_theta_prev[lay] = self.d_theta[lay]

            # TODO: Feed-forward all training patterns and calculate their prediction quadratic error

            # TODO: Feed-forward all validation patterns and calculate their prediction quadratic error

        # TODO: Feed-forward all test patterns

        return ''

    # X an array of size (n_samples,n_features) that contains the samples.
    # This method returns a vector with the predicted values for all the input samples
    def predict(self, X):

        return

    # that returns 2 arrays of size (n_epochs, 2) that contain the evolution of the training error and the validation
    # error for each of the epochs of the system, so this information can be plotted.
    def loss_epochs(self):

        return


def sigmoid_derivative(x):
    return x * (1 - x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def linear(x):
    return x


def tanh(x):
    return np.tanh(x)


# Function calls
activation_functions = {
    'sigmoid': sigmoid,
    'relu': relu,
    'linear': linear,
    'tanh': tanh
}


# Call the 'switch'
def activate(activation_function, x):
    if activation_function in activation_functions:
        return activation_functions[activation_function]

# Better made switch
def activate_derivative(activation_function, x):
    if activation_function == 'sigmoid':
        return x * (1 - x)
    elif activation_function == 'relu':
        return np.where(x > 0, 1, 0)
    elif activation_function == 'linear':
        return np.ones_like(x)
    elif activation_function == 'tanh':
        return 1 - np.tanh(x)**2

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

dataSynth = pd.read_csv('synthetic_Normalized.txt', sep='\t')
dataSynthTEST = pd.read_csv('synthetic_Normalized_TEST.txt', sep='\t')

X_s = dataSynth[:, :-1]  # All except last column
y_s = dataSynth[:, -1]  # last column
X_s_TEST = dataSynthTEST[:, :-1]  # All except last column
y_s_TEST = dataSynthTEST[:, -1]  # last column

dataTurb = pd.read_csv('turbine_Standardized.txt', sep='\t')
dataTurbTEST = pd.read_csv('turbine_Standardized_TEST.txt', sep='\t')

X_t = dataTurb[:, :-1]  # All except last column
y_t = dataTurb[:, -1]  # last column
X_t_TEST = dataTurbTEST[:, :-1]  # All except last column
y_t_TEST = dataTurbTEST[:, -1]  # last column

layers = [4, 9, 5, 1]  # layers include input layer + hidden layers + output layer

nn = MyNeuralNetwork(layers, 1000, 0.01, 0.0, 'sigmoid', '')  # Creation nn

nn.fit(X_s, y_s)  # Training

prediction = nn.predict(X_s_TEST)  # Making prediction

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
