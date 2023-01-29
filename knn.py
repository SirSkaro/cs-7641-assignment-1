import data_utils
from data_utils import Task

from sklearn.neighbors import KNeighborsClassifier

### Docs used:
# https://scikit-learn.org/stable/modules/neighbors.html
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


def learn(task: Task):
    training_set, test_set = data_utils.get_training_and_test_sets(task)
    classifier = KNeighborsClassifier(
        n_neighbors=3,
        weights='uniform',
        p=2, # Euclidiean metric used
        metric='minkowski',
        algorithm='brute'
    )

    classifier.fit(training_set.samples, training_set.labels)

    error = 1.0 - classifier.score(test_set.samples, test_set.labels)

    return classifier, error
