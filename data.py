import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


# def generate_data0(*, n=500, p=50, m1=0, m2=1):
#     x1 = np.random.multivariate_normal(m1 * np.ones([p]), np.eye(p), n)
#     x2 = np.random.multivariate_normal(m2 * np.ones([p]), np.eye(p), n)
#     x = np.concatenate([x1, x2], axis=0)
#     y = np.concatenate([np.ones([n]), -np.ones([n])])
#     return x, y


def generate_data0(seed=131):
    np.random.seed(seed)
    X, Y = make_classification(n_samples=1000, n_features=2, n_redundant=0, class_sep=2)
    Y = 2 * Y - 1
    # plt.scatter(*X.T, c=Y)
    # plt.show()
    return train_test_split(X, Y, random_state=123, test_size=0.2)


# # feature selection
# def generate_data1(*, n=500, k=10, p=50):
#     X = np.concatenate([np.random.multivariate_normal(np.zeros([p]), np.eye(p), size=n)])
#     X_sum = np.sum(X[:, :k]*X[:, :k], axis=1)
#     y = -np.ones([n])
#     y[X_sum > sc.stats.chi2.isf(0.5, k)] = 1
#     return X, y
#
#
# # feature selection
# def generate_data2(*, n=500, k=10, p=50):
#     X = np.concatenate([np.random.multivariate_normal(np.zeros([p]), np.eye(p), size=n)])
#     X_sum = np.sum(np.abs(X[:, :k]), axis=1)
#     y = -np.ones([n])
#     y[X_sum > k] = 1
#     return X, y


@dataclass
class Dataset:
    X_train: np.array
    X_test: np.array
    y_train: np.array
    y_test: np.array
    name: str


# class GeneratedDataset(Dataset):
#     generate_functions = [generate_data0, generate_data1, generate_data2]
#
#     def __init__(self, dataset_number=0, n=500, p=50):
#         self.X_train, self.y_train = self.generate_functions[dataset_number](n=n, p=p)
#         self.X_test, self.y_test = self.generate_functions[dataset_number](n=n, p=p)
#         self.name = self.generate_functions[dataset_number].__name__


def load_banknote_data():
    banknote = pd.read_csv("data/banknote.csv")
    permutation = np.arange(0, banknote.shape[0])
    np.random.shuffle(permutation)
    banknote = banknote.iloc[permutation, :]
    y = np.array(banknote['Class'])
    y[y == 1] = -1
    y[y == 2] = 1
    X = np.array(banknote.drop(columns='Class'))
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, random_state=123, test_size=0.2)


def load_breast_cancer_data():
    breast_cancer = load_breast_cancer()
    X = StandardScaler().fit_transform(breast_cancer.data)
    y = breast_cancer.target
    permutation = np.arange(0, X.shape[0])
    np.random.shuffle(permutation)
    X = X[permutation, :]
    y = y[permutation]
    y[y == 0] = -1
    return train_test_split(X, y, random_state=123, test_size=0.2)


class RealDataset(Dataset):
    functions = [generate_data0, load_banknote_data, load_breast_cancer_data]
    names = ['artificial', 'banknote', 'breast cancer']

    def __init__(self, dataset_number=0):
        self.X_train, self.X_test, self.y_train, self.y_test = self.functions[dataset_number]()
        self.name = self.names[dataset_number]


np.random.seed(123)
#
datasets = [RealDataset(0), RealDataset(1), RealDataset(2)]
