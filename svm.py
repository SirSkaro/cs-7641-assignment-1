import data_utils
from data_utils import Task, SampleSet

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

### Docs used:
# https://scikit-learn.org/stable/modules/svm.html#svm
# https://scikit-learn.org/stable/modules/svm.html#svm-kernels
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC


# TODO Use decision tree to find the 3 best feature and create a 3D graph
# https://stackoverflow.com/questions/51495819/how-to-plot-svm-decision-boundary-in-sklearn-python

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


def plot(clf, training_set: SampleSet):
    # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
    # Solve for w3 (z)
    z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]
    X, Y = training_set.samples, training_set.labels

    tmp = np.linspace(-5, 5, 30)
    x, y = np.meshgrid(tmp, tmp)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], 'ob')
    ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], 'sr')
    ax.plot_surface(x, y, z(x, y))
    ax.view_init(30, 60)
    plt.show()
