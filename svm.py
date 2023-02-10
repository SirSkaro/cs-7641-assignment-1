import data_utils
from data_utils import Task, SampleSet

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from time import time

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
        kernel=RADIAL_BASIS_FUNCTION,
        gamma='auto',
        C=5.0,    # small = underfit, large = overfit
        decision_function_shape=ONE_VS_ONE,
        random_state=0,
        cache_size=2000  # in MB
    )

    classifier.fit(training_set.samples, training_set.labels)
    error = 1.0 - classifier.score(test_set.samples, test_set.labels)
    return classifier, error


def weighted_shuffle(task: Task, kernel: str, C: float, percent_training: float = 0.90):
    training_set, test_set = data_utils.get_training_and_test_sets(task, percent_training=percent_training, randomize=True)
    classifier = SVC(
        kernel=kernel,
        C=C,
        gamma='auto',
        degree=5,
        class_weight='balanced',
        decision_function_shape=ONE_VS_ONE,
        random_state=0,
        cache_size=2000  # in MB
    )

    classifier.fit(training_set.samples, training_set.labels)
    error = 1.0 - classifier.score(test_set.samples, test_set.labels)
    train_error = 1.0 - classifier.score(training_set.samples, training_set.labels)
    return classifier, error, train_error


def iterate_weighted_shuffle(task: Task, percent_training: float = 0.90):
    kernels = [LINEAR, POLYNOMIAL, RADIAL_BASIS_FUNCTION]
    Cs = [5, 20, 100]
    candidate_classifiers = {
        LINEAR: (None, 1.0, 1.0),
        POLYNOMIAL: (None, 1.0, 1.0),
        RADIAL_BASIS_FUNCTION: (None, 1.0, 1.0)
    }

    for C in Cs:
        for kernel in kernels:
            print(f'Training "{kernel}" kernel with C={C}...')
            start = time()
            classifier, error, train_error = weighted_shuffle(task, kernel, C, percent_training)
            end = time()
            training_time = round(end - start, 2)
            print(f'\tFinished in {training_time} seconds with error {error}')
            if error < candidate_classifiers[kernel][1]:
                candidate_classifiers[kernel] = (classifier, error, train_error, training_time)
                print(f'\tNew best classifier for kernel')

    return candidate_classifiers


def create_learning_curve():
    kernels = [LINEAR, POLYNOMIAL, RADIAL_BASIS_FUNCTION]
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
        test_errors = {
            LINEAR: [],
            POLYNOMIAL: [],
            RADIAL_BASIS_FUNCTION: []
        }
        train_errors = {
            LINEAR: [],
            POLYNOMIAL: [],
            RADIAL_BASIS_FUNCTION: []
        }
        training_times = {
            LINEAR: [],
            POLYNOMIAL: [],
            RADIAL_BASIS_FUNCTION: []
        }
        for percent_training in percentages:
            classifiers = iterate_weighted_shuffle(task, percent_training=percent_training)
            for kernel in kernels:
                test_errors[kernel].append(classifiers[kernel][1])
                train_errors[kernel].append(classifiers[kernel][2])
                training_times[kernel].append(classifiers[kernel][3])

        for kernel in kernels:
            ax[0].plot(percentages, test_errors[kernel], label=f'{name} Test Error - {kernel.upper()}', marker="o", drawstyle="steps-post", linestyle=linestyle)
            ax[0].plot(percentages, train_errors[kernel], label=f'{name} Train Error - {kernel.upper()}', marker="o", drawstyle="steps-post", linestyle=linestyle)
            ax[1].plot(percentages, training_times[kernel], label=f'{name} Classifier - {kernel.upper()}', marker="o", drawstyle="steps-post", linestyle=linestyle)

    ax[0].legend(loc="best")
    ax[1].legend(loc="best")
