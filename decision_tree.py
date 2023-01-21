import data_utils
from data_utils import Task

from sklearn import tree
import numpy as np


### Docs used:
# https://scikit-learn.org/stable/modules/tree.html
# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py

def learn(task: Task):
    training_set, test_set = data_utils.get_training_and_test_sets(task)
    classifier = tree.DecisionTreeClassifier().fit(training_set.samples, training_set.labels)

    correctly_classified_count = np.sum(test_set.labels == classifier.predict(test_set.samples))
    error = 1.0 - (correctly_classified_count / test_set.size())

    return classifier, error
