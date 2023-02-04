import data_utils
from data_utils import Task, SampleSet

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import distance_metrics
import numpy as np

from typing import Tuple

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


def expectation_maximization(task: Task, error_increase_streak: int = 5):
    training_set, test_set = data_utils.get_training_and_test_sets(task)
    previous_k = None
    current_k = 1
    previous_metric = None
    current_metric = USABLE_METRICS[0]
    candidate_classifier = None
    error = 1

    while not ( (current_k == previous_k) and (current_metric == previous_metric) ):
        previous_metric = current_metric
        previous_k = current_k
        current_metric = iterate_metric(task, current_k, (training_set, test_set))[0].metric
        candidate_classifier, error = iterate_k(task, current_metric, datasets=(training_set, test_set), streak=error_increase_streak)
        current_k = candidate_classifier.n_neighbors
        print(f'Finished iteration. '
              f'\n\tPrevious/current metric: {previous_metric}/{current_metric}'
              f'\n\tPrevious/current k: {previous_k}/{current_k}')

    return candidate_classifier, error

