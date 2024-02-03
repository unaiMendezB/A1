import numpy as np
import pandas as pd

# Neural Network class
class MyNeuralNetwork:
    def __init__(self, layers, epochs, lr, momentum, activation, validation_set_percentage):

        self.L = len(layers)
        self.n = layers
        self.epochs = epochs
        self.learning_rate = lr
        self.momentum = momentum
        self.validation_set_percentage = validation_set_percentage
        self.fact = activation
        self.w = [np.random.randn(self.n[i], self.n[i-1]) for i in range(1, self.L)]
        self.theta = [np.random.randn(ni, 1) for ni in self.n[1:]]
        self.d_w_prev = [np.zeros((self.n[i], self.n[i-1])) for i in range(1, self.L)]
        self.d_theta_prev = [np.zeros((ni, 1)) for ni in self.n[1:]]
        self.loss_train = []
        self.loss_val = []

        self.xi = []  # node values
        for lay in range(self.L):
            self.xi.append(np.zeros(layers[lay]))

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


    def forward_propagation(self, x):
        self.h = [x]
        self.xi = []
        for l in range(self.L - 1):
            self.xi.append(np.dot(self.w[l], self.h[-1]) + self.theta[l])
            self.h.append(self.activate(self.xi[-1]))
        return self.h[-1]

    def backward_propagation(self, y):
        self.delta = [self.h[-1] - y]
        for l in range(self.L - 2, -1, -1):
            self.delta.append(np.dot(self.w[l].T, self.delta[-1]) * self.activate_derivative(self.xi[l]))
        self.delta = self.delta[::-1]

    def update_weights(self):
        self.d_w = [self.lr * np.dot(self.delta[l], self.h[l].T) + self.momentum * self.d_w_prev[l] for l in range(self.L - 1)]
        self.d_theta = [self.lr * dl + self.momentum * dt for dl, dt in zip(self.delta[1:], self.d_theta_prev)]
        self.w = [wl - dwl for wl, dwl in zip(self.w, self.d_w)]
        self.theta = [tl - dtl for tl, dtl in zip(self.theta, self.d_theta)]
        self.d_w_prev = self.d_w
        self.d_theta_prev = self.d_theta

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


    # This method allows us to train the network with this data.
    # X array of size (n_samples,n_features) which holds the training samples represented
    # as floating point feature vectors
    # y of size (n_samples), which holds the target values (class labels) for the training samples.
    def fit(self, X, y):
        X_train, X_val, y_train, y_val = self.split_dataset(X, y)
        for _ in range(self.epochs):
            idx = np.random.permutation(X_train.shape[0])
            for i in idx:
                self.forward_propagation(X_train[i])
                self.backward_propagation(y_train[i])
                self.update_weights()
            self.loss_train.append(self.loss(y_train, self.predict(X_train)))
            self.loss_val.append(self.loss(y_val, self.predict(X_val)))

    def predict(self, X):
        return np.array([self.forward_propagation(x) for x in X])

    def loss_epochs(self):
        return np.array([self.loss_train, self.loss_val]).T


# dataTurb = pd.read_csv('turbine_Standardized.txt', sep='\t')
dataSynth = pd.read_csv('synthetic_Normalized.txt', sep='\t')
X = dataSynth.iloc[:, :-1].values  # Create the array X
y = dataSynth.iloc[:, -1].values  # Create the vector y

layers = [4, 9, 5, 1]  # layers include input layer + hidden layers + output layer

nn = MyNeuralNetwork(layers, 100, 0.01, 0.9, 'sigmoid', 0.2,)  # Creation nn
nn.fit(X, y)  # Training
train_errors, val_errors = nn.loss_epochs()

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
