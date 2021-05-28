from classifier import (
    CoordinateClassifier, PermutedCoordinateClassifier, OnlineCoordinateClassifier,
    Classifier, CMLSClassifier, TrustRegionNewtonClassifier)
from metric import Metric
from data import datasets, Dataset
import pandas as pd
import time


def quality_of_classifier_on_dataset(classifier: Classifier, dataset: Dataset, max_iter: int):
    start = time.time()
    c = classifier(max_iter=max_iter).fit(dataset.X_train, dataset.y_train)
    y_pred = c._predict(dataset.X_test)
    return {
        'Acc': Metric.Acc.evaluate(dataset.y_test, y_pred),
        'F1': Metric.F1score.evaluate(dataset.y_test, y_pred),
        'Precision': Metric.Precision.evaluate(dataset.y_test, y_pred),
        'Recall': Metric.Recall.evaluate(dataset.y_test, y_pred),
        'Time': time.time() - start,
    }


# c_iter, p_iter, o_iter = 4, 4, 18
# c_iter, p_iter, o_iter = 20, 20, 20
# CMLSClassifier
classifiers = [TrustRegionNewtonClassifier, CoordinateClassifier, PermutedCoordinateClassifier, OnlineCoordinateClassifier]
max_iter = 20
results = {}
i = 1
n = len(datasets) * len(classifiers)
for dataset in datasets:
    for classifier in classifiers:
        results[f'{dataset.name}, {classifier.__name__}'] = quality_of_classifier_on_dataset(classifier, dataset, max_iter)
        print(f'Ended {i}/{n}: {dataset.name}, {classifier.__name__}')

df = pd.DataFrame(results).T

df[['Acc', 'Precision', 'Recall']]
