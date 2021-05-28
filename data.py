import numpy as np
import scipy as sc
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


def generate_data0(*, n=500, p=50, m1=0, m2=1):
    x1 = np.random.multivariate_normal(m1 * np.ones([p]), np.eye(p), n)
    x2 = np.random.multivariate_normal(m2 * np.ones([p]), np.eye(p), n)
    x = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([np.ones([n]), -np.ones([n])])
    return x, y


# feature selection
def generate_data1(*, n=500, k=10, p=50):
    X = np.concatenate([np.random.multivariate_normal(np.zeros([p]), np.eye(p), size=n)])
    X_sum = np.sum(X[:, :k]*X[:, :k], axis=1)
    y = -np.ones([n])
    y[X_sum > sc.stats.chi2.isf(0.5, 10)] = 1
    return X, y


# feature selection
def generate_data2(*, n=500, k=10, p=50):
    X = np.concatenate([np.random.multivariate_normal(np.zeros([p]), np.eye(p), size=n)])
    X_sum = np.sum(np.abs(X[:, :k]), axis=1)
    y = -np.ones([n])
    y[X_sum > k] = 1
    return X, y


@dataclass
class Dataset:
    X_train: np.array
    X_test: np.array
    y_train: np.array
    y_test: np.array
    name: str


class GeneratedDataset(Dataset):
    generate_functions = [generate_data0, generate_data1, generate_data2]

    def __init__(self, dataset_number=0, n=500, p=50):
        self.X_train, self.y_train = self.generate_functions[dataset_number](n=n, p=p)
        self.X_test, self.y_test = self.generate_functions[dataset_number](n=n, p=p)
        self.name = self.generate_functions[dataset_number].__name__


def load_banknote_data():
    banknote = pd.read_csv("data/banknote.csv")
    permutation = np.arange(0, banknote.shape[0])
    np.random.shuffle(permutation)
    banknote = banknote.iloc[permutation, :]
    y = np.array(banknote['Class'])
    y[y == 1] = -1
    y[y == 2] = 1
    X = np.array(banknote.drop(columns='Class'))
    return train_test_split(X, y, random_state=123, test_size=0.2)


def load_diabetes_data():
    diabetes = pd.read_csv("data/diabetes.csv")
    permutation = np.arange(0, diabetes.shape[0])
    np.random.shuffle(permutation)
    diabetes = diabetes.iloc[permutation, :]
    y = np.array(diabetes['Outcome'])
    y[y == 0] = -1
    X = np.array(diabetes.drop(columns='Outcome'))
    return train_test_split(X, y, random_state=123, test_size=0.2)


class RealDataset(Dataset):
    real_functions = [load_banknote_data, load_diabetes_data]

    def __init__(self, dataset_number=0):
        self.X_train, self.X_test, self.y_train, self.y_test = self.real_functions[dataset_number]()
        self.name = self.real_functions[dataset_number].__name__


np.random.seed(123)
datasets = [GeneratedDataset(0), GeneratedDataset(1), GeneratedDataset(2), RealDataset(0), RealDataset(1)]
