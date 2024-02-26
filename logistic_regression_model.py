import numpy
import numpy as np


class logistic_model():
    def __init__(self, inp_features):
        self.weights = np.zeros((inp_features, 1), dtype=float)
        self.weights = self.weights.reshape(self.weights.shape[0], 1)
        self.b = 0.0
        self.losses = []

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(Z))
        return A

    def forward(self, X):
        #X is a concated x's horizontally

        Z = np.dot(self.weights.T, X) + self.b
        A = self.sigmoid(Z)

        return A

    def train(self, X, Y):
        learning_rate = 1e-2
        epochs = 50000
        m = X.shape[1]

        for i in range(epochs):
            A = self.forward(X)

            dw = np.dot(X, (A - Y).T) / m
            db = np.sum(A - Y) / m

            self.weights = self.weights - learning_rate * dw
            self.b  = self.b - learning_rate * db
            if i == 0:
                self.losses.append(self.cost(X, Y))
            if i % 100 == 0:
                self.losses.append(self.cost(X, Y))

    def cost(self, X, Y):
        Y_hat = self.forward(X)
        m = X.shape[1]
        losses = -(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

        return np.sum(losses) / m
