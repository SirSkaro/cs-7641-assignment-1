import data_utils
from data_utils import Task

from sklearn import tree


### Docs used:
# https://scikit-learn.org/stable/modules/tree.html
# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py

def learn(task: Task):
    dataset, labels = data_utils.parse_data(task)
    training_set = dataset[:20000]
    test_set = dataset[20000:]

    classifier = tree.DecisionTreeClassifier()
    return classifier.fit(training_set, labels)
