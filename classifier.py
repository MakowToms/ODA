from stopper import Stopper
from metric import Metric
import numpy as np
from scipy.optimize import minimize


class Classifier:
    def __init__(self, name='Classifier', C=1, **kwargs):
        self.C = C
        self.stopper = Stopper(**kwargs)
        self.w = None
        self.n = None
        self.p = None
        self.X = None
        self.y = None
        self.name = name

    def fit(self, X, y):
        X = self._add_ones(X)
        y = y.reshape(-1, 1)
        self.n = y.shape[0]
        self.p = X.shape[1]
        self.w = np.zeros([self.p, 1])
        self.X = X
        self.y = y
        while not self.stopper.stop(self):
            # print(Metric.Acc.evaluate(self._predict(X), y))
            self._train_outer_iteration()
            # y_pred_proba = self._predict(X)
            # self.log_likelihood.append(self._log_likelihood(y, y_pred_proba))
        return self

    def predict(self, X):
        X = self._add_ones(X)
        return self._predict(X)

    def score(self, X, y_true, metric: Metric = Metric.Acc):
        y_pred = self.predict(X)
        return metric.evaluate(y_true, y_pred)

    def _train_outer_iteration(self):
        pass

    def _predict(self, X):
        """
        :param X: matrix with observations: n_observations x n_predictors
        :return: predictions as np.array n_observations x 1
        """
        res = np.sign(X @ self.w)
        res = res.reshape(-1)
        res[res == 0] = 1  # should be rare unless w==0
        return res

    def _add_ones(self, X):
        ones = np.ones([X.shape[0], 1])
        return np.concatenate([ones, X], axis=1)

    def L2_SVM_loss(self, w=None):
        if w is None:
            w = self.w
        loss = 0
        for i in range(self.n):
            loss += max(1 - self.y[i, 0] * self.X[i, :] @ w, 0) ** 2
        loss *= self.C
        return loss + w.T @ w / 2

    @staticmethod
    def _log_likelihood(y_true, y_pred_proba):
        return (np.log(y_pred_proba + 1e-6).T @ y_true + np.log(1 - y_pred_proba + 1e-6).T @ (1 - y_true))[0, 0]


class CoordinateClassifier(Classifier):

    def __init__(self, name='Coordinate', sigma=0.01, Beta=0.5, **kwargs):
        self.sigma = sigma
        self.Beta = Beta
        super().__init__(name=name, **kwargs)

    def _train_outer_iteration(self):
        for i in range(self.p):
            self._train_inner_iteration(i)

    def _train_inner_iteration(self, i):
        d = - self.compute_D_derivative(i, 0) / self.compute_D_second_derivative(i, 0)
        lambda_ = 1
        j = 0
        while not self.optimally_constraint_12(i, lambda_*d):
            j += 1
            if j >= 1000:
                print('Ended because of too many iterations')
                break
            lambda_ *= self.Beta
        self.w[i, 0] += lambda_*d

    def compute_D(self, i, z):
        e = np.zeros([self.p, 1])
        e[i, 0] = 1
        return self.L2_SVM_loss(self.w + z*e)

    def compute_D_derivative(self, i, z):
        e = np.zeros([self.p, 1])
        e[i, 0] = 1
        w = self.w + z*e
        derivative = 0
        for j in range(self.n):
            derivative += max(1 - self.y[j, 0] * self.X[j, :] @ w, 0) * self.y[j] * self.X[j, i]
        derivative *= (-2 * self.C)
        return derivative + w[i, 0]

    def compute_D_second_derivative(self, i, z):
        e = np.zeros([self.p, 1])
        e[i, 0] = 1
        w = self.w + z * e
        derivative = 0
        for j in range(self.n):
            derivative += np.sign(max(1 - self.y[j, 0] * self.X[j, :] @ w, 0)) * (self.X[j, i] ** 2)
        derivative *= (2 * self.C)
        return derivative + 1

    # constraint 12 from paper
    def optimally_constraint_12(self, i, z):
        return self.compute_D(i, z) - self.compute_D(i, 0) <= - self.sigma * (z**2)


class PermutedCoordinateClassifier(CoordinateClassifier):

    def __init__(self, name='Permuted', **kwargs):
        super().__init__(name=name, **kwargs)

    def _train_outer_iteration(self):
        coordinates = np.random.choice(self.p, self.p, replace=False)
        for coordinate in coordinates:
            self._train_inner_iteration(coordinate)


class OnlineCoordinateClassifier(CoordinateClassifier):

    def __init__(self, name='Online', **kwargs):
        super().__init__(name=name, **kwargs)

    def _train_outer_iteration(self):
        coordinate = np.random.randint(0, self.p)
        self._train_inner_iteration(coordinate)


class TrustRegionNewtonClassifier(CoordinateClassifier):
    def __init__(self, name='TrustRegion', eta0=1e-4, eta1=0.25, eta2=0.75, sigma1=0.25, sigma2=0.5, sigma3=4, **kwargs):
        super().__init__(name=name, **kwargs)

        if eta0 <= 0:
            raise ValueError("eta0 must be greater than 0")

        if eta2 >= 1:
            raise ValueError("eta2 must be lesser than 1")

        if eta1 <= 0:
            raise ValueError("eta1 must be greater than 0")

        if eta2 <= eta1:
            raise ValueError("eta2 must be greater than eta1")

        if sigma3 <= 1:
            raise ValueError("sigma3 must be greater than 1")

        if sigma1 <= 0:
            raise ValueError("sigma1 must be greater than 0")

        if sigma2 >= 1:
            raise ValueError("sigma2 must be lesser than 1")

        if sigma2 <= sigma1:
            raise ValueError("sigma2 must be greater than sigma1")

        self.eta0 = eta0
        self.eta1 = eta1
        self.eta2 = eta2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3

        self.__delta = 1

    def _train_outer_iteration(self):
        d = np.array([self.compute_D_derivative(i, 0)[0] for i in range(self.p)])
        d2 = np.array([self.compute_D_second_derivative(i, 0)[0] for i in range(self.p)])
        q = lambda s: (d*s + 0.5 * s**2 * d2).sum()
        s = minimize(
            fun=q,
            x0=np.zeros(self.p), method='trust-ncg',
            jac=lambda s: s*d2 + d,
            hess=lambda s: np.diag(d2)).x

        if (s**2).sum() > self.__delta:
            s *= np.sqrt(self.__delta / (s**2).sum())

        ro = (self.L2_SVM_loss(self.w + s.reshape(-1, 1)) - self.L2_SVM_loss()) / q(s)

        if ro > self.eta0:
            self.w += s.reshape(-1, 1)

        if ro <= self.eta1:
            self.__delta = np.random.uniform(
                self.sigma1*min((s**2).sum(), self.__delta),
                self.sigma2*self.__delta)
        elif self.eta1 < ro < self.eta2:
            self.__delta = np.random.uniform(
                self.sigma1*self.__delta, self.sigma3*self.__delta
            )
        else:
            self.__delta = np.random.uniform(
                self.__delta, self.sigma3*self.__delta
            )

class CMLSClassifier(CoordinateClassifier):
    def __init__(self, name='CMLS', **kwargs):
        super().__init__(name=name, **kwargs)
        self.__outer_iter = 0

        self.__delta = None

    def fit(self, X, y):
        self.__delta = np.ones(X.shape[1]+1) * 10
        # self.z = np.zeros(X.shape[1]+1)
        super(CMLSClassifier, self).fit(X, y)
        return self

    def _train_outer_iteration(self):
        self.ck = max(0, 1 - self.__outer_iter/50)
        # self.w_old = np.array(self.w)
        for i in range(self.p):
            self._train_inner_iteration(i)
        self.__outer_iter += 1

    def beta(self, w):
        ywx = (np.matmul(self.X, w) * self.y).reshape(-1,)
        return np.where(1 - ywx > 0, 1 - ywx, self.ck*(1 - ywx))

    def beta2(self, w, i):
        return np.where((np.matmul(self.X, w) * self.y).reshape(-1,) <= 1 + np.abs(self.__delta[i]*self.X[:, i]), 2*self.C, 0)

    def _train_inner_iteration(self, i):
        # wz = np.array(self.w_old)
        # wz[i] += self.z[i]
        beta = self.beta2(self.w, i)
        U = 1 + beta.sum()
        d = self.compute_D_derivative(i, 0)

        z = min(max(-d/U, -self.__delta[i]), self.__delta[i])

        self.__delta[i] = 2*np.abs(z) + 1e-1
        self.w[i] += z

# wykres liniowy L2 lossu od iteracji
# czas na logarytmiczny y, datasets x classifiers
