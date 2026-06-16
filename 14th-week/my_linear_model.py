import numpy as np

# fit, predict

# 시험 대비:
# multiple 변수 쓰는 것도 만들어보기
class MySimpleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, Y):
        x = X[:, 0]
        y = Y[:, 0]
        x_mean = X.mean()
        y_mean = Y.mean()
        self.coef_ = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        self.intercept_ = y_mean - self.coef_ * x_mean

    def predict(self, X):
        x = X[:, 0]
        return (self.intercept_ + self.coef_ * x).reshape(-1, 1)
