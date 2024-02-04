import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Neural Network class
class MyNeuralNetwork:
    def __init__(self, layers, epochs, lr, momentum, activation, validation_set_percentage):

        # Number of layers
        self.L = len(layers)

        # Number of neurons in each layer
        self.n = layers

        # Field values
        self.h = []
        for lay in range(self.L):
            self.h.append(np.zeros(layers[lay]))

        # Node values
        self.xi = []
        for lay in range(self.L):
            self.xi.append(np.zeros(layers[lay]))

        # edge weights
        self.w = []
        self.w.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.w.append(np.zeros((layers[lay], layers[lay - 1])))

        # Threshold values
        self.theta = []
        for lay in range(self.L):
            self.theta.append(np.zeros(layers[lay]))

        # Propagation of errors
        self.delta = []
        for lay in range(self.L):
            self.delta.append(np.zeros(layers[lay]))

        # Array of matrices for the changes of the weights
        self.d_w = []
        self.d_w.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.d_w.append(np.zeros((layers[lay], layers[lay - 1])))

        # Array of arrays for the changes of the weights
        self.d_theta = []
        for lay in range(self.L):
            self.d_theta.append(np.zeros(layers[lay]))

        # Previous changes of the weights, used for the momentum term
        self.d_w_prev = []
        self.d_w_prev.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.d_w_prev.append(np.zeros((layers[lay], layers[lay - 1])))

        # Previous changes of the thresholds, used for the momentum term
        self.d_theta_prev = []
        for lay in range(self.L):
            self.d_theta_prev.append(np.zeros(layers[lay]))

        # Activation function
        self.fact = activation

        # The number of epochs for the training process
        self.epochs = epochs

        # The learning rate for the training process
        self.learning_rate = lr

        # The momentum term for the training process
        self.momentum = momentum

        # The percentage of the dataset to be used as a validation set
        self.validation_set_percentage = validation_set_percentage

        # A list to store the loss for the training set after each epoch
        self.loss_train = []

        # A list to store the loss for the validation set after each epoch
        self.loss_val = []

    # Separates the data to train from the one to Test
    def split_dataset(self, X, y):
        num_test = int(self.validation_set_percentage * len(y))

        # Shuffle the data (for turb or synt we dont need this but it can be helpfull for other datasets)
        indices = np.random.permutation(len(y))

        # Split the data
        X_train = X[indices[num_test:]]
        y_train = y[indices[num_test:]]
        X_test = X[indices[:num_test]]
        y_test = y[indices[:num_test]]

        return X_train, X_test, y_train, y_test

    # Switch for activation_function
    def activate(self, x):
        if self.fact == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.fact == 'relu':
            return np.maximum(0, x)
        elif self.fact == 'linear':
            return x
        elif self.fact == 'tanh':
            return np.tanh(x)

    # Switch for activation_derivate_function
    def activate_derivative(self, x):
        if self.fact == 'sigmoid':
            return x * (1 - x)
        elif self.fact == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.fact == 'linear':
            return np.ones_like(x)
        elif self.fact == 'tanh':
            return 1 - np.tanh(x) ** 2

    def fit(self, X, y):
        # 1. Scale input and/or output patterns, if needed
        X_train, X_test, y_train, y_test = self.split_dataset(X, y)

        # 2. Initialize all weights and thresholds randomly
        for lay in range(1, self.L):
            self.w[lay] = np.random.randn(self.n[lay], self.n[lay - 1])
            self.theta[lay] = np.random.randn(self.n[lay])

        # 3. For epoch = 1 To num epochs
        for epoch in range(self.epochs):
            # 4. For pat = 1 To num training patterns
            for pat in range(X_train.shape[0]):
                # 5. Choose a random pattern (x, z) of the training set
                idx = np.random.randint(0, X_train.shape[0])
                x = X_train[idx]
                z = y_train[idx]

                # 6. Feed-forward propagation of pattern x to obtain the output o(x)
                self.forward_propagation(x)

                # 7. Back-propagate the error for this pattern
                self.back_propagate(z)

                # 8. Update the weights and thresholds
                self.update_weights_and_thresholds()

            # 10. Feed-forward all training patterns and calculate their prediction quadratic error
            self.loss_train.append(self.loss(X_train, y_train))

            # 11. Feed-forward all validation patterns and calculate their prediction quadratic error
            self.loss_val.append(self.loss(X_test, y_test))

        # 13. Plot the evolution of the training and validation errors
        self.plot_errors()

    def forward_propagation(self, x):
        self.xi[0] = x
        for lay in range(1, self.L):
            self.h[lay] = np.dot(self.w[lay], self.xi[lay - 1]) - self.theta[lay]
            self.xi[lay] = self.activate(self.h[lay])

    def back_propagate(self, z):
        self.delta[-1] = (z - self.xi[-1]) * self.activate_derivative(self.h[-1])
        for lay in range(self.L - 2, -1, -1):  # For the hidden layers
            self.delta[lay] = self.activate_derivative(self.h[lay]) * np.dot(self.w[lay + 1].T, self.delta[lay + 1])

    def update_weights_and_thresholds(self):
        for lay in range(1, self.L):
            self.d_w[lay] = self.learning_rate * np.outer(self.delta[lay], self.xi[lay - 1]) + self.momentum * \
                            self.d_w_prev[lay]
            self.d_theta[lay] = -self.learning_rate * self.delta[lay] + self.momentum * self.d_theta_prev[lay]
            self.w[lay] += self.d_w[lay]
            self.theta[lay] += self.d_theta[lay]
            self.d_w_prev[lay] = self.d_w[lay]
            self.d_theta_prev[lay] = self.d_theta[lay]

    def loss(self, X, y):
        loss = 0
        for i in range(X.shape[0]):
            self.forward_propagation(X[i])
            loss += np.sum((y[i] - self.xi[-1]) ** 2)  # Quadratic error
        return loss / X.shape[0]  # Mean quadratic error

    def plot_errors(self):
        plt.plot(range(self.epochs), self.loss_train, label='Training loss')
        plt.plot(range(self.epochs), self.loss_val, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            self.forward_propagation(X[i])
            predictions.append(self.xi[-1])  # The output of the network is the prediction
        return np.array(predictions)

    def loss_epochs(self):
        # Combine the training and validation losses into a single array
        loss_epochs = np.zeros((self.epochs, 2))
        loss_epochs[:, 0] = self.loss_train
        loss_epochs[:, 1] = self.loss_val
        return loss_epochs


# dataTurb = pd.read_csv('turbine_Standardized.txt', sep='\t')
dataSynth = pd.read_csv('synthetic_Normalized.txt', sep='\t')
X = dataSynth.iloc[:, :-1].values  # Create the array X
y = dataSynth.iloc[:, -1].values  # Create the vector y

layers = [4, 9, 5, 1]  # layers include input layer + hidden layers + output layer

nn = MyNeuralNetwork(layers, 100, 0.01, 0.9, 'sigmoid', 0.2, )  # Creation nn
nn.fit(X, y)  # Training
train_errors, val_errors = nn.loss_epochs()
