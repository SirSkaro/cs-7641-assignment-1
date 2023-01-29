import data_utils
from data_utils import Task

from sklearn.svm import SVC

### Docs used:
# https://scikit-learn.org/stable/modules/svm.html#svm
# https://scikit-learn.org/stable/modules/svm.html#svm-kernels
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC


# Classification strategy
ONE_VS_ONE = 'ovo'
ONE_VS_REST = 'ovr'

# Kernel functions
LINEAR = 'linear'
POLYNOMIAL = 'poly'
RADIAL_BASIS_FUNCTION = 'rbf'
SIGMOID = 'sigmoid'


def learn(task: Task):
    training_set, test_set = data_utils.get_training_and_test_sets(task)
    classifier = SVC(
        kernel=POLYNOMIAL,
        degree=3,   # degree of the polynomial - ignored by all other kernels
        gamma='auto',
        tol=1e-3,
        decision_function_shape=ONE_VS_ONE,
        random_state=0
    )

    classifier.fit(training_set.samples, training_set.labels)

    error = 1.0 - classifier.score(test_set.samples, test_set.labels)

    return classifier, error
