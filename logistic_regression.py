import numpy as np
import itertools
import math

def squared_l2_norm(w):

    l2 = np.linalg.norm(w,ord=2)**2
    return l2

def binary_cross_entropy(y_hat, y):

    n = y_hat.shape[0]
    bce = (1/n)*(y * np.log(y_hat+1e-8) + (1 - y) * np.log(1 - y_hat+1e-8))
    return bce

def sigmoid(x):

    sig = 1/(1 + np.exp(-x))
    return sig

def accuracy(y_pred, y):

    acc = ((y_pred == y).sum())/y_pred.size
    return acc

def calculate_batches(X, batch_size):

    indices = list(range(X.shape[0]))
    np.random.shuffle(indices)
    args = [iter(indices)] * batch_size
    batch_indices = itertools.zip_longest(*args, fillvalue=None)
    return [list(j for j in i if j is not None) for i in batch_indices]

class LogisticRegression(object):
    def __init__(self, input_dimensions=2, seed=1234):

        np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self._initialize_weights()

    def _initialize_weights(self):

        self.weights = np.random.randn(self.input_dimensions+1,1)

    def fit(self, X_train, y_train, X_val, y_val, num_epochs=10, batch_size=4, alpha=0.01, _lambda=0.0):

        train_xent = [] # append your cross-entropy on training set to this after each epoch
        val_xent = [] # append your cross-entropy on validation set to this after each epoch
        batch_indices = calculate_batches(X_train, batch_size)

        for epoch in range(num_epochs):
            for batch in batch_indices:
                self._train_on_batch(X_train[batch], y_train[batch], alpha, _lambda)
            y_hat_train = self.predict_proba(X_train)
            y_hat_val = self.predict_proba(X_val)
            train_xent.append(binary_cross_entropy(y_hat_train, y_train))
            val_xent.append(binary_cross_entropy(y_hat_val, y_val))
            return (train_xent, val_xent)

        return (train_xent, val_xent)

    def predict_proba(self, X):

        return sigmoid(np.matmul(X,self.weights))

    def predict(self, X):

        predict = np.where(sigmoid(np.matmul(X,self.weights)) >= 0.5, 1,0)
        return predict

    def _train_on_batch(self, X, y, alpha, _lambda):

        tob = alpha*np.add(self._binary_cross_entropy_gradient(X,y), _lambda*self._l2_regularization_gradient())
        self.weights = np.subtract(self.weights,tob)

    def _binary_cross_entropy_gradient(self, X, y):


        return np.matmul(X.T,np.subtract(self.predict_proba(X),y))/ self.input_dimensions

    def _l2_regularization_gradient(self):

        return self.weights

if __name__ == "__main__":
    print("This is library code. You may implement some functions here to do your own debugging if you want, but they won't be graded")