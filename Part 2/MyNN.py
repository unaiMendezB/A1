import numpy as np

class MyNeuralNetwork:
    def __init__(self, layers, units, epochs, lr, momentum, activation, validation_split):
        self.L = layers
        self.n = units
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.validation_split = validation_split
        self.fact = activation
        self.w = [np.random.randn(self.n[i], self.n[i-1]) for i in range(1, self.L)]
        self.theta = [np.random.randn(ni, 1) for ni in self.n[1:]]
        self.d_w_prev = [np.zeros((self.n[i], self.n[i-1])) for i in range(1, self.L)]
        self.d_theta_prev = [np.zeros((ni, 1)) for ni in self.n[1:]]
        self.loss_train = []
        self.loss_val = []

    def activation(self, z):
        if self.fact == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.fact == 'relu':
            return np.maximum(0, z)
        elif self.fact == 'linear':
            return z
        elif self.fact == 'tanh':
            return np.tanh(z)

    def activation_derivative(self, z):
        if self.fact == 'sigmoid':
            return self.activation(z) * (1 - self.activation(z))
        elif self.fact == 'relu':
            return (z > 0).astype(int)
        elif self.fact == 'linear':
            return 1
        elif self.fact == 'tanh':
            return 1 - np.tanh(z)**2

    def forward_propagation(self, x):
        self.h = [x]
        self.xi = []
        for l in range(self.L - 1):
            self.xi.append(np.dot(self.w[l], self.h[-1]) + self.theta[l])
            self.h.append(self.activation(self.xi[-1]))
        return self.h[-1]

    def backward_propagation(self, y):
        self.delta = [self.h[-1] - y]
        for l in range(self.L - 2, -1, -1):
            self.delta.append(np.dot(self.w[l].T, self.delta[-1]) * self.activation_derivative(self.xi[l]))
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

    def fit(self, X, Y):
        val_size = int(X.shape[0] * self.validation_split)
        X_train, X_val = X[:-val_size], X[-val_size:]
        Y_train, Y_val = Y[:-val_size], Y[-val_size:]
        for _ in range(self.epochs):
            idx = np.random.permutation(X_train.shape[0])
            for i in idx:
                self.forward_propagation(X_train[i])
                self.backward_propagation(Y_train[i])
                self.update_weights()
            self.loss_train.append(self.loss(Y_train, self.predict(X_train)))
            self.loss_val.append(self.loss(Y_val, self.predict(X_val)))

    def predict(self, X):
        return np.array([self.forward_propagation(x) for x in X])

    def loss_epochs(self):
        return np.array([self.loss_train, self.loss_val]).T
