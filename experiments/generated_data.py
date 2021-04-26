from classifier import CoordinateClassifier, PermutedCoordinateClassifier
import numpy as np
from metric import Metric


def generate_data(m1, m2, n, p):
    x1 = np.random.multivariate_normal(m1 * np.ones([p]), np.eye(p), n)
    x2 = np.random.multivariate_normal(m2 * np.ones([p]), np.eye(p), n)
    x = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([np.ones([n]), -np.ones([n])])
    return x, y


n = 100
np.random.seed(123)
x, y = generate_data(0, 1, n, 2)
c_iter, p_iter = 10, 25
c = CoordinateClassifier(max_iter=c_iter).fit(x, y)
p = PermutedCoordinateClassifier(max_iter=p_iter).fit(x, y)
print(f'Coordinate classifier with {c_iter} iterations - accuracy = {Metric.Acc.evaluate(y, c._predict(x))}')
print(f'Permuted coordinate classifier with {p_iter} iterations - accuracy = {Metric.Acc.evaluate(y, p._predict(x))}')
print('Hypothesis: permuted descent needs less total number of iterations and can achieve better results')