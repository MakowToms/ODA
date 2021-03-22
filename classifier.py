from stopper import Stopper
from metric import Metric
import numpy as np


class Classifier:
    def __init__(self, **kwargs):
        self.stopper = Stopper(**kwargs)
        self.beta = None

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        self.beta = np.zeros([X.shape[1], 1])
        while not self.stopper.stop(self):
            self._train_iteration(X, y)
            y_pred_proba = self._predict(X)
            self.log_likelihood.append(self._log_likelihood(y, y_pred_proba))
        return self

    def predict_proba(self, X):
        return self._predict(X).reshape(-1)

    def predict(self, X, threshold=0.5):
        y_pred = np.zeros(X.shape[0])
        y_pred[self._predict(X).reshape(-1) > threshold] = 1
        return y_pred

    def score(self, X, y_true, metric: Metric):
        y_pred = self.predict(X)
        return metric.evaluate(y_true, y_pred)

    def _train_iteration(self, X, y):
        p = self._predict(X)
        self.beta += self._compute_derivative(X, y, p)

    def _compute_derivative(self, X, y, p):
        pass

    def _predict(self, X):
        """
        :param X: matrix with observations: n_observations x n_predictors
        :return: predictions as np.array n_observations x 1
        """
        return 1 / (1 + np.exp(-X @ self.beta))

    @staticmethod
    def _log_likelihood(y_true, y_pred_proba):
        return (np.log(y_pred_proba).T @ y_true + np.log(1 - y_pred_proba).T @ (1 - y_true))[0, 0]


class IRLS(Classifier):

    def __init__(self, eps=0.001, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def _compute_derivative(self, X, y, p):
        W = np.diag([x*(1-x) + self.eps for x in p.reshape(-1)])
        return np.linalg.inv(X.T @ W @ X) @ X.T @ (y.reshape((-1, 1)) - p.reshape(-1, 1))


