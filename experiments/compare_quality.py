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


def params_of_coordinate_descent(dataset: Dataset, max_iter: int, sigma=0.01, Beta=0.5, C=1):
    c = CoordinateClassifier(sigma=sigma, Beta=Beta, C=C, max_iter=max_iter).fit(dataset.X_train, dataset.y_train)
    return c.stopper.loss_history, c.stopper.time_history


max_iter = 200
results = {}
i = 1
n = len(datasets) * 4
for dataset in datasets:
    loss_histories = []
    time_histories = []
    classifier_names = []

    for sigma in [0.001, 0.01, 0.1, 0.5]:
        loss_history, time_history = params_of_coordinate_descent(dataset, max_iter, sigma=sigma)
        print(f'Ended {i}/{n}: {dataset.name}, {sigma}')
        print(loss_history)
        i += 1
        loss_histories.append(loss_history)
        time_histories.append(time_history)
        classifier_names.append(sigma)

    # loss plot
    for i in range(4):
        plt.plot(loss_histories[i], label=classifier_names[i])
    plt.title(f'Training loss for {dataset.name} dataset, compare sigma')
    plt.xlabel('iterations')
    plt.ylabel('L2 SVM loss')
    plt.yscale('log')
    plt.yticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dataset.name}_sigma_iters.png', pad_inches=0.2)
    plt.show()

    for i in range(4):
        plt.plot(time_histories[i], loss_histories[i], label=classifier_names[i])
        print(classifier_names[i])
        print(time_histories[i])
    plt.title(f'Training loss for {dataset.name} dataset, compare sigma')
    plt.xlabel('time')
    plt.ylabel('L2 SVM loss')
    plt.yscale('log')
    plt.yticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dataset.name}_sigma_time.png', pad_inches=0.2)
    plt.show()


max_iter = 200
results = {}
i = 1
n = len(datasets) * 4
for dataset in datasets:
    loss_histories = []
    time_histories = []
    classifier_names = []

    for Beta in [0.01, 0.1, 0.5, 0.9]:
        loss_history, time_history = params_of_coordinate_descent(dataset, max_iter, Beta=Beta)
        print(f'Ended {i}/{n}: {dataset.name}, {Beta}')
        print(loss_history)
        i += 1
        loss_histories.append(loss_history)
        time_histories.append(time_history)
        classifier_names.append(Beta)

    # loss plot
    for i in range(4):
        plt.plot(loss_histories[i], label=classifier_names[i])
    plt.title(f'Training loss for {dataset.name} dataset, compare Beta')
    plt.xlabel('iterations')
    plt.ylabel('L2 SVM loss')
    plt.yscale('log')
    plt.yticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dataset.name}_Beta_iters.png', pad_inches=0.2)
    plt.show()

    for i in range(4):
        plt.plot(time_histories[i], loss_histories[i], label=classifier_names[i])
        print(classifier_names[i])
        print(time_histories[i])
    plt.title(f'Training loss for {dataset.name} dataset, compare Beta')
    plt.xlabel('time')
    plt.ylabel('L2 SVM loss')
    plt.yscale('log')
    plt.yticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dataset.name}_Beta_time.png', pad_inches=0.2)
    plt.show()


max_iter = 200
results = {}
i = 1
n = len(datasets) * 4
for dataset in datasets:
    loss_histories = []
    time_histories = []
    classifier_names = []

    for C in [0.01, 0.1, 1, 10]:
        loss_history, time_history = params_of_coordinate_descent(dataset, max_iter, C=C)
        print(f'Ended {i}/{n}: {dataset.name}, {C}')
        print(loss_history)
        i += 1
        loss_histories.append(loss_history)
        time_histories.append(time_history)
        classifier_names.append(C)

    # loss plot
    for i in range(4):
        plt.plot(loss_histories[i], label=classifier_names[i])
    plt.title(f'Training loss for {dataset.name} dataset, compare C')
    plt.xlabel('iterations')
    plt.ylabel('L2 SVM loss')
    plt.yscale('log')
    plt.yticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dataset.name}_C_iters.png', pad_inches=0.2)
    plt.show()

    for i in range(4):
        plt.plot(time_histories[i], loss_histories[i], label=classifier_names[i])
        print(classifier_names[i])
        print(time_histories[i])
    plt.title(f'Training loss for {dataset.name} dataset, compare C')
    plt.xlabel('time')
    plt.ylabel('L2 SVM loss')
    plt.yscale('log')
    plt.yticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dataset.name}_C_time.png', pad_inches=0.2)
    plt.show()
