import numpy as np

# KNN (K-Nearest-Neighbors)
# - 가까운 점 k개의 클래스를 확인해서 가장 개수가 많은 클래스를 선택한다.
#  - 데이터 테이블의 각 row가 벡터가 되고, 벡터간의 거리 구한다.
# - Regression에서도 쓸 수 있다. 가장 가까운 k개 점의 평균을 내면...

class MyKNeighborsClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X = None
        self.Y = None
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.n_classes = len(np.unique(Y))
    
    def predict(self, X):
        X_train = self.X.reshape(1, self.X.shape[0], -1)
        X_test = X.reshape(X.shape[0], 1, -1)
        dist = np.sqrt(((X_train - X_test) ** 2).sum(axis=2))
        sorted_idx = dist.argsort(axis=1)[:, :self.n_neighbors]
        Y_pred = self.Y[sorted_idx]
        uniques = np.unique(Y_pred)
        counts = (Y_pred[:, :, np.newaxis] == uniques).sum(axis=1)
        Y_pred = uniques[counts.argmax(axis=1)]
        return Y_pred

class MyKNeighborsRegressor:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X = None
        self.Y = None
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.n_classes = len(np.unique(Y))
    
    def predict(self, X):
        X_train = self.X.reshape(1, self.X.shape[0], -1)
        X_test = X.reshape(X.shape[0], 1, -1)
        dist = np.sqrt(((X_train - X_test) ** 2).sum(axis=2))
        sorted_idx = dist.argsort(axis=1)[:, :self.n_neighbors]
        Y_pred = self.Y[sorted_idx].mean(axis=1)
        return Y_pred