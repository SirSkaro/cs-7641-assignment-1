import data_utils
from data_utils import Task
from decision_tree import statistics as tree_statistics
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import ClassifierMixin


### Docs used:
# https://scikit-learn.org/stable/modules/ensemble.html#adaboost
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-multiclass-py

def learn(task: Task, shuffle: bool = False, weak_classifier: ClassifierMixin = None, ensemble_size: int = 50):
    training_set, test_set = data_utils.get_training_and_test_sets(task, randomize=shuffle)
    classifier = AdaBoostClassifier(
        n_estimators=ensemble_size,
        estimator=weak_classifier,
        algorithm='SAMME.R',
        learning_rate=1.0,
        random_state=0
    )

    classifier.fit(training_set.samples, training_set.labels)

    error = 1.0 - classifier.score(test_set.samples, test_set.labels)

    return classifier, error


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
