import data_utils
from data_utils import Task

from sklearn import tree
import numpy as np
import graphviz


### Docs used:
# https://scikit-learn.org/stable/modules/tree.html
# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py

def round1(task: Task):
    training_set, test_set = data_utils.get_training_and_test_sets(task)
    classifier = tree.DecisionTreeClassifier().fit(training_set.samples, training_set.labels)

    correctly_classified_count = np.sum(test_set.labels == classifier.predict(test_set.samples))
    error = 1.0 - (correctly_classified_count / test_set.size())

    return classifier, error


def vizualize(task: Task, round_function):
    classifier, error = round_function(task)
    filename = f'graphs/{task.name} - {round_function.__name__}'
    export = tree.export_graphviz(classifier, out_file=None,
                                    feature_names=task.value.features,
                                    class_names=task.value.classes,
                                    filled=True, rounded=True,
                                    special_characters=True,
                                    impurity=True)
    graph = graphviz.Source(export)
    graph.render(filename)
