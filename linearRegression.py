"""
Pablo Valencia A01700912
@coronapl
Linear Regression Framework
March 27, 2023
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate, num_iters):
        self.__weights = None
        self.__bias = None
        self.__alpha = learning_rate
        self.__epochs = num_iters
        self.__errors = []

    def __get_cost(self, x_data, y_data):
        return self.predict(x_data) - y_data

    def __gradient_descend(self, x_data, y_data):
        m = x_data.shape[0]
        errors = self.__get_cost(x_data, y_data)

        d_weights = (1 / m) * np.dot(x_data.T, errors)
        d_bias = (1 / m) * np.sum(errors)

        self.__weights -= self.__alpha * d_weights
        self.__bias -= self.__alpha * d_bias

        return errors

    def get_params(self):
        return self.__weights, self.__bias

    def get_errors(self):
        return len(self.__errors)

    def fit(self, x_train, y_train):
        self.__weights = np.random.rand(x_train.shape[1], 1)
        self.__bias = np.random.randn()

        for i in range(self.__epochs):
            errors = self.__gradient_descend(x_train, y_train)
            self.__errors.append(np.mean(errors))

    def predict(self, x_data):
        predictions = np.dot(x_data, self.__weights) + self.__bias
        return predictions

    def plot(self):
        plt.plot([i for i in range(1, len(self.__errors) + 1)], self.__errors)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Model Error by Training Iteration')
        plt.show()
