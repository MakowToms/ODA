from classifier import (
    CoordinateClassifier, PermutedCoordinateClassifier, OnlineCoordinateClassifier,
    Classifier, CMLSClassifier, TrustRegionNewtonClassifier)
from metric import Metric
from data import datasets, Dataset
import pandas as pd
import time
import matplotlib.pyplot as plt


def quality_of_classifier_on_dataset(classifier: Classifier, dataset: Dataset, max_iter: int):
    start = time.time()
    c = classifier(max_iter=max_iter).fit(dataset.X_train, dataset.y_train)
    y_pred = c.predict(dataset.X_test)
    return {
        'Accuracy': Metric.Acc.evaluate(dataset.y_test, y_pred),
        'F1': Metric.F1score.evaluate(dataset.y_test, y_pred),
        'Precision': Metric.Precision.evaluate(dataset.y_test, y_pred),
        'Recall': Metric.Recall.evaluate(dataset.y_test, y_pred),
        'Time': time.time() - start,
    }, c.stopper.loss_history, c.stopper.time_history


classifiers = [CMLSClassifier, TrustRegionNewtonClassifier, CoordinateClassifier, PermutedCoordinateClassifier, OnlineCoordinateClassifier]
max_iter = 200
results = {}
i = 1
n = len(datasets) * len(classifiers)
times = {}
for dataset in datasets:
    dataset_times = {}
    dataset_results = {}
    loss_histories = []
    time_histories = []
    classifier_names = []

    for classifier in classifiers:
        quality, loss_history, time_history = quality_of_classifier_on_dataset(classifier, dataset, max_iter)
        dataset_times[classifier().name] = quality['Time']
        results[f'{dataset.name}, {classifier().name}'] = quality
        print(quality)
        print(f'Ended {i}/{n}: {dataset.name}, {classifier().name}')
        i += 1
        dataset_results[f'{classifier().name}'] = quality
        loss_histories.append(loss_history)
        time_histories.append(time_history)
        classifier_names.append(classifier().name)

    times[dataset.name] = dataset_times

    # loss plot
    for i in range(5):
        plt.plot(loss_histories[i], label=classifier_names[i])
    plt.title(f'Training loss for {dataset.name} dataset')
    plt.xlabel('iterations')
    plt.ylabel('L2 SVM loss')
    plt.yscale('log')
    plt.yticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    plt.legend()
    plt.show()

    # time loss plot
    for i in range(5):
        plt.plot(time_histories[i], loss_histories[i], label=classifier_names[i])
        print(classifier_names[i])
        print(time_histories[i])
    plt.title(f'Training loss for {dataset.name} dataset')
    plt.xlabel('time')
    plt.ylabel('L2 SVM loss')
    plt.yscale('log')
    plt.yticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    plt.legend()
    plt.show()

    # quality plot
    df = pd.DataFrame(dataset_results)
    # df = df - 0.9
    df.iloc[:4, :].plot(kind='bar')  # , bottom=0.9
    plt.legend(loc='lower right')
    plt.title(f'Quality comparison for {dataset.name} dataset')
    plt.ylabel('measure value')
    plt.xticks(rotation=0)
    plt.show()

df = pd.DataFrame(times).T
df.plot(kind='bar')
plt.legend(loc='lower right')
plt.yscale('log')
plt.title(f'Time comparison of algorithms')
plt.ylabel('time')
plt.yticks([1, 3, 10, 30], ['1', '3', '10', '30'])
plt.xticks(rotation=0)
plt.show()
