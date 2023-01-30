import data_utils
from data_utils import Task

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import graphviz
import matplotlib.pyplot as plt


### Docs used:
# https://scikit-learn.org/stable/modules/tree.html
# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
# https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

def basic(task: Task):
    training_set, test_set = data_utils.get_training_and_test_sets(task)
    classifier = DecisionTreeClassifier(random_state=0)

    classifier.fit(training_set.samples, training_set.labels)

    correctly_classified_count = np.sum(test_set.labels == classifier.predict(test_set.samples))
    error = 1.0 - (correctly_classified_count / test_set.size())

    return classifier, error


def pruning(task: Task):
    training_set, test_set = data_utils.get_training_and_test_sets(task)
    base_classifier = DecisionTreeClassifier(random_state=0)

    # Get candidate alpha values for pruning
    path = base_classifier.cost_complexity_pruning_path(training_set.samples, training_set.labels)
    candidate_alphas = np.array(path.ccp_alphas)
    print(f'Found {len(candidate_alphas)} candidate alphas')
    candidate_alphas = candidate_alphas.round(decimals=4)
    candidate_alphas = np.unique(candidate_alphas)
    print(f'Trimmed to {len(candidate_alphas)} candidate alphas')

    # Train pruned classifiers
    pruned_classifiers = []
    for candidate_alpha in candidate_alphas:
        print(f'Fitting decision tree for alpha {candidate_alpha}...')
        pruned_classifier = DecisionTreeClassifier(random_state=0, ccp_alpha=candidate_alpha)
        pruned_classifier.fit(training_set.samples, training_set.labels)
        pruned_classifiers.append(pruned_classifier)
        print(f'Finished fitting tree for alpha {candidate_alpha} with {pruned_classifier.tree_.node_count} nodes '
              f'and max depth {pruned_classifier.tree_.max_depth}')

    # Visualize number of nodes vs alpha
    node_counts = [classifiers.tree_.node_count for classifiers in pruned_classifiers]
    depth = [classifier.tree_.max_depth for classifier in pruned_classifiers]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(candidate_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("Alpha")
    ax[0].set_ylabel("# of Node")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(candidate_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("Alpha")
    ax[1].set_ylabel("Tree depth")
    ax[1].set_title("Depth vs Alpha")
    fig.savefig(f'graphs/decision tress/{task.name} - nodes+depth vs alpha.png')

    # Visualize accuracy
    print('Scoring trees')
    train_scores = [clf.score(training_set.samples, training_set.labels) for clf in pruned_classifiers]
    test_scores = [clf.score(test_set.samples, test_set.labels) for clf in pruned_classifiers]
    print('Finished scoring trees')
    print(f'\t Train scores: {train_scores}')
    print(f'\t Train scores: {test_scores}')

    fig, ax = plt.subplots()
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Alpha for Training and Testing Sets")
    ax.plot(candidate_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(candidate_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    fig.savefig(f'graphs/decision tress/{task.name} - accuracy.png')

    # return best classifier and error
    best_index = np.argmax(test_scores)
    best_classifier = pruned_classifiers[best_index]
    error = 1 - test_scores[best_index]

    return best_classifier, error


def vizualize(task: Task, round_function):
    classifier, error = round_function(task)
    filename = f'graphs/decision trees/{task.name} - {round_function.__name__}'
    export = tree.export_graphviz(classifier, out_file=None,
                                  feature_names=task.value.features,
                                  class_names=task.value.classes,
                                  filled=True, rounded=True,
                                  special_characters=True,
                                  impurity=True)
    graph = graphviz.Source(export)
    graph.render(filename)
