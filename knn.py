import data_utils
from data_utils import Task, SampleSet

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import distance_metrics
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from time import time

### Docs used:
# https://scikit-learn.org/stable/modules/neighbors.html
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

USABLE_METRICS = list(distance_metrics().keys())
USABLE_METRICS.remove('haversine')
USABLE_METRICS.remove('precomputed')


def learn(task: Task, k: int = 3):
    training_set, test_set = data_utils.get_training_and_test_sets(task)
    classifier = KNeighborsClassifier(
        n_neighbors=k,
        weights='uniform',
        p=2, # Euclidiean metric used
        metric='minkowski',
        algorithm='brute'
    )

    classifier.fit(training_set.samples, training_set.labels)

    error = 1.0 - classifier.score(test_set.samples, test_set.labels)

    return classifier, error


def iterate_metric(task: Task, k: int = 3, datasets: Tuple[SampleSet, SampleSet] = None):
    if datasets is None:
        training_set, test_set = data_utils.get_training_and_test_sets(task)
    else:
        training_set, test_set = datasets[0], datasets[1]
    candidate_classifiers = []

    for metric in USABLE_METRICS:
        print(f'Training using metric {metric}')
        candidate_classifier = KNeighborsClassifier(
            n_neighbors=k,
            weights='uniform',
            p=2,
            metric=metric,
            algorithm='brute'
        )
        candidate_classifier.fit(training_set.samples, training_set.labels)
        error = 1.0 - candidate_classifier.score(test_set.samples, test_set.labels)
        candidate_classifiers.append((candidate_classifier, error))
        print(f'\tMetric {metric} has an error of {error}')

    return sorted(candidate_classifiers, key=lambda classifier_error_pair: classifier_error_pair[1])[0]


def iterate_k(task: Task, metric: str, datasets: Tuple[SampleSet, SampleSet] = None, streak: int = 5):
    if datasets is None:
        training_set, test_set = data_utils.get_training_and_test_sets(task)
    else:
        training_set, test_set = datasets[0], datasets[1]
    candidate_classifiers = []
    test_errors = []

    for k in range(1, training_set.size()):
        print(f'Training using k={k}')
        candidate_classifier = KNeighborsClassifier(
            n_neighbors=k,
            weights='uniform',
            p=2,
            metric=metric,
            algorithm='brute'
        )
        candidate_classifier.fit(training_set.samples, training_set.labels)
        error = 1.0 - candidate_classifier.score(test_set.samples, test_set.labels)
        candidate_classifiers.append((candidate_classifier, error))
        test_errors.append(error)
        print(f'\tk={k} has an error of {error}')

        # stopping condition: check for certain number of iterations and monotonically increasing
        if len(test_errors) % streak == 0 and np.all(np.diff(test_errors[-streak:]) > 0):
            break

    return sorted(candidate_classifiers, key=lambda classifier_error_pair: classifier_error_pair[1])[0]


def expectation_maximization(task: Task,
                             initial_k: int = 10,
                             error_increase_streak: int = 5,
                             percent_training: float = 0.9,
                             percent_validation: float = 0):

    if percent_validation > 0:
        training_set, validation_set, test_set = data_utils.get_training_validation_and_test_sets(task,
                                                                                      percent_training=percent_training,
                                                                                      percent_validation=percent_validation,
                                                                                      randomize=True)
    else:
        training_set, test_set = data_utils.get_training_and_test_sets(task, percent_training=percent_training, randomize=True)
        validation_set = test_set

    previous_k = None
    current_k = initial_k
    previous_metric = None
    current_metric = USABLE_METRICS[0]
    candidate_classifier = None

    while not ( (current_k == previous_k) and (current_metric == previous_metric) ):
        previous_metric = current_metric
        previous_k = current_k
        current_metric = iterate_metric(task, current_k, (training_set, validation_set))[0].metric
        candidate_classifier = iterate_k(task, current_metric, datasets=(training_set, validation_set), streak=error_increase_streak)[0]
        current_k = candidate_classifier.n_neighbors
        print(f'Finished iteration. '
              f'\n\tPrevious/current metric: {previous_metric}/{current_metric}'
              f'\n\tPrevious/current k: {previous_k}/{current_k}')

    error = 1.0 - candidate_classifier.score(test_set.samples, test_set.labels)
    return candidate_classifier, error


def best_expectation_maximization_shuffle(task: Task,
                                            iterations: int = 5,
                                            initial_k: int = 10,
                                            error_increase_streak: int = 5,
                                            percent_training: float = 0.9,
                                            percent_validation: float = 0):
    candidate_classifiers = []
    for iteration in range(iterations):
        print(f'Starting iteration {iteration}')
        candidate_classifier, error = expectation_maximization(task,
                                                               initial_k=initial_k,
                                                               percent_training=percent_training,
                                                               percent_validation=percent_validation,
                                                               error_increase_streak=error_increase_streak)
        candidate_classifiers.append((candidate_classifier, error))

    print(f'Candidate classifiers are {candidate_classifiers}')
    return sorted(candidate_classifiers, key=lambda classifier_error_pair: classifier_error_pair[1])[0]


def create_learning_curve():
    fig, ax = plt.subplots(1, 2)

    ax[0].set_title("Learning Curve")
    ax[0].set_xlabel("Percentage Training Set")
    ax[0].set_ylabel("Error")

    ax[1].set_title("Training Time")
    ax[1].set_xlabel("Percentage Training Set")
    ax[1].set_ylabel("Training Time (in Seconds)")

    percentages = np.linspace(0, 1, 11)[1:-1]
    for task, name, linestyle in [(Task.SCRIBE_RECOGNITION, 'Scribe', 'dotted'),
                                  (Task.LETTER_RECOGNITION, 'Letter', 'dashed')]:
        test_errors = []
        train_errors = []
        training_times = []
        configs = []
        for percent_training in percentages:
            start = time()
            classifier, test_error = best_expectation_maximization_shuffle(task, iterations=5, error_increase_streak=5, percent_training=percent_training, percent_validation=0.0)
            end = time()

            test_errors.append(test_error)
            train_errors.append(0.0)
            training_times.append(round(end - start, 2))
            configs.append((classifier.n_neighbors, classifier.metric))

        ax[0].plot(percentages, test_errors, label=f'{name} Test Error', marker="o", drawstyle="steps-post",
                   linestyle=linestyle)
        ax[0].plot(percentages, train_errors, label=f'{name} Train Error', marker="o", drawstyle="steps-post",
                   linestyle=linestyle)
        ax[1].plot(percentages, training_times, label=f'{name} Classifier', marker="o", drawstyle="steps-post",
                   linestyle=linestyle)

        #for index, classifer in enumerate(classifiers):
        #    ax[0].annotate(f'k={classifer.n_neighbors}\nmetric={classifer.metric}', (percentages[index], test_error[index]))

    ax[0].legend(loc="best")
    ax[1].legend(loc="best")

    return ax, test_errors, configs
