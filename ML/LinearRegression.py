import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


# 均方误差MSE
def mean_squared_error(y_predicted, y_true):
    return np.mean((y_predicted - y_true) ** 2)


# 决定系数R2
def r_square(y_predicted, y_true):
    s = np.sum((y_predicted - y_true) ** 2)
    m = np.sum((np.mean(y_predicted) - y_true) ** 2)
    return 1 - s / m


# 校正决定系数
def r_square_adjusted(y_predicted, y_true, X):
    s = (1 - r_square(y_predicted, y_true)) * (y_true.size - 1)
    m = y_true.size - X.shape[1] - 1
    return 1 - s / m


def draw():
    mses = []
    accuracies = []
    for i in range(50):
        X, y = datasets.make_regression(n_samples=100, n_features=1, noise=i, random_state=5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        test_predicted = lr.predict(X_test)
        MSE = mean_squared_error(test_predicted, y_test)
        acc = r_square(test_predicted, y_test)
        mses.append(MSE)
        accuracies.append(acc)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('noise')
    ax1.set_ylabel('mse')
    ax1.plot(range(50), mses, color='red')
    ax2 = ax1.twinx()
    ax2.set_ylabel('acc')
    ax2.plot(range(50), accuracies, color='blue')
    fig.tight_layout()
    plt.show()

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weight = np.zeros(n_features)
        self.bias = 0

        # 利用梯度下降更新w, b
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weight) + self.bias
            # 计算w和 b的梯度
            dw = (1 / n_samples) * np.dot(X.T, y_predicted - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weight) + self.bias
        return y_predicted


if __name__ == '__main__':
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    test_predicted = lr.predict(X_test)
    MSE = mean_squared_error(test_predicted, y_test)
    accuracy = r_square(test_predicted, y_test) * 100
    r2_adjusted = r_square_adjusted(test_predicted, y_test, X_test) * 100
    print(f'均方误差为{MSE:.3f}')
    print(f'准确率为{accuracy:.2f}%')
    print(f'{r2_adjusted:.2f}')
    draw()

