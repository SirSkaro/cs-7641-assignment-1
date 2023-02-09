import data_utils
from data_utils import Task
from decision_tree import statistics as tree_statistics
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from time import time

### Docs used:
# https://scikit-learn.org/stable/modules/ensemble.html#adaboost
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-multiclass-py


def learn(task: Task, shuffle: bool = False, weak_classifier: ClassifierMixin = None,
          ensemble_size: int = 50, percent_training: float = 0.9):
    training_set, test_set = data_utils.get_training_and_test_sets(task,
                                                                   percent_training=percent_training,
                                                                   randomize=shuffle)
    classifier = AdaBoostClassifier(
        n_estimators=ensemble_size,
        estimator=weak_classifier,
        algorithm='SAMME.R',
        learning_rate=1.0,
        random_state=0
    )

    classifier.fit(training_set.samples, training_set.labels)

    error = 1.0 - classifier.score(test_set.samples, test_set.labels)
    train_error = 1.0 - classifier.score(training_set.samples, training_set.labels)

    return classifier, error, train_error


def statistics(clf: AdaBoostClassifier):
    node_counts = []
    depth_counts = []
    leaf_counts = []
    median_leaf_depths = []
    for estimator in clf.estimators_:
        print(f'Estimator {len(node_counts)} | {str(estimator)}')
        n_nodes, max_depth, n_leaves, median_leaf_depth = tree_statistics(estimator)
        node_counts.append(n_nodes)
        depth_counts.append(max_depth)
        leaf_counts.append(n_leaves)
        median_leaf_depths.append(median_leaf_depth)

    median_node_count = np.median(node_counts)
    median_depth = np.median(depth_counts)
    median_leaf_count = np.median(leaf_counts)
    median_leaf_depth = np.median(median_leaf_depths)

    print(f'Median Nodes: {median_node_count} | '
          f'Median Max Depth: {median_depth} | '
          f'Median Leaf Nodes: {median_leaf_count} | '
          f'Median leaf depth: {median_leaf_depth}')


def create_learning_curve():
    fig, ax = plt.subplots(1, 2)

    ax[0].set_title("Learning Curve")
    ax[0].set_xlabel("Percentage Training Set")
    ax[0].set_ylabel("Error")

    ax[1].set_title("Training Time")
    ax[1].set_xlabel("Percentage Training Set")
    ax[1].set_ylabel("Training Time (in Seconds)")

    percentages = np.linspace(0, 1, 11)[1:-1]
    for task, name, linestyle, alpha in [(Task.SCRIBE_RECOGNITION, 'Scribe', 'dotted', 0.000005),
                                         (Task.LETTER_RECOGNITION, 'Letter', 'dashed', 0.0000005)]:
        test_errors = []
        train_errors = []
        training_times = []
        weak_classifier = DecisionTreeClassifier(max_depth=20, ccp_alpha=alpha)
        for percent_training in percentages:
            start = time()
            _, test_error, train_error = learn(task, percent_training=percent_training,
                                               shuffle=True, weak_classifier=weak_classifier,
                                               ensemble_size=100)
            end = time()

            test_errors.append(test_error)
            train_errors.append(train_error)
            training_times.append(round(end - start, 2))

        ax[0].plot(percentages, test_errors, label=f'{name} Test Error', marker="o", drawstyle="steps-post", linestyle=linestyle)
        ax[0].plot(percentages, train_errors, label=f'{name} Train Error', marker="o", drawstyle="steps-post", linestyle=linestyle)
        ax[1].plot(percentages, training_times, label=f'{name} Classifier', marker="o", drawstyle="steps-post", linestyle=linestyle)

    ax[0].legend(loc="best")
    ax[1].legend(loc="best")
