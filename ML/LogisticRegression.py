import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


def accuracy(y_predicted, y_true):
    return np.sum(y_predicted == y_true) / y_true.size


def _sigmoid(x):
    x_ravel = x.ravel()
    y = []
    for i in range(len(x_ravel)):
        val = x_ravel[i]
        if val >= 0:
            y.append(1 / (1 + np.exp(-val)))
        else:
            y.append(np.exp(val) / 1 + np.exp(val))
    return np.array(y).reshape(x.shape)


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weight) + self.bias
            y_predicted = _sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, y_predicted - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weight) + self.bias
        predicted = _sigmoid(linear_model)
        return predicted


if __name__ == '__main__':
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1255)
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    test_predicted = logistic_regression.predict(X_test)
    acc = accuracy(test_predicted, y_test)
    print(f'准确率为{acc*100:.2f}%')

