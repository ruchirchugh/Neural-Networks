import numpy as np
import itertools

def squared_l2_norm(w):
    l2 = np.linalg.norm(w, ord=2) ** 2
    return l2


def mean_squared_error(y_hat, y):

    mse = (np.square(y_hat - y)).mean(axis=0)
    return mse

def calculate_batches(X, batch_size):

    indices = list(range(X.shape[0]))
    np.random.shuffle(indices)
    args = [iter(indices)] * batch_size
    batch_indices = itertools.zip_longest(*args, fillvalue=None)
    return [list(j for j in i if j is not None) for i in batch_indices]


class LinearRegression(object):
    def __init__(self, input_dimensions=2, seed=1234):

        np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self._initialize_weights()

    def _initialize_weights(self):

        self.weights = np.random.randn(self.input_dimensions + 1, 1)

    def fit(self, X_train, y_train, X_val, y_val, num_epochs=10, batch_size=16, alpha=0.01, _lambda=0.0):

        train_error = []  # append your MSE on training set to this after each epoch
        val_error = []  # append your MSE on validation set to this after each epoch

        batch_indices = calculate_batches(X_train, batch_size)
        for epoch in range(num_epochs):
            for batch in batch_indices:
                train_error = self._train_on_batch(X_train[batch], y_train[batch], alpha, _lambda)

            val_error = self._train_on_batch(X_val, y_val, alpha, _lambda)

        return train_error, val_error

    def predict(self, X):

        return np.matmul(X, self.weights)

    def _train_on_batch(self, X, y, alpha, _lambda):

        tob = alpha * np.add(self._mse_gradient(X, y), _lambda * self._l2_regularization_gradient())
        self.weights = np.subtract(self.weights, tob)
        return tob

    def _mse_gradient(self, X, y):

        mse_g = np.matmul(X.T, np.subtract(self.predict(X), y)) / self.input_dimensions
        return mse_g

    def _l2_regularization_gradient(self):

        l2_reg = 2 * self.weights / self.input_dimensions
        return l2_reg


if __name__ == "__main__":
    print("This is library code. You may implement some functions here to do your own debugging if you want, but they won't be graded")