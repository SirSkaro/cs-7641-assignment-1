import data_utils
from data_utils import Task

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import graphviz
import matplotlib.pyplot as plt
from time import time


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


def pruning(task: Task, percent_training: float = 0.9, shuffle: bool = False):
    training_set, test_set = data_utils.get_training_and_test_sets(task, percent_training, shuffle)
    base_classifier = DecisionTreeClassifier(random_state=0, criterion='entropy')

    # Get candidate alpha values for pruning
    path = base_classifier.cost_complexity_pruning_path(training_set.samples, training_set.labels)
    candidate_alphas = np.array(path.ccp_alphas)
    print(f'Found {len(candidate_alphas)} candidate alphas')
    candidate_alphas = candidate_alphas.round(decimals=6)
    candidate_alphas = np.unique(candidate_alphas)
    print(f'Trimmed to {len(candidate_alphas)} candidate alphas')

    # Train pruned classifiers
    pruned_classifiers = []
    train_scores = []
    test_scores = []
    for candidate_alpha in candidate_alphas:
        print(f'Fitting decision tree for alpha {candidate_alpha}...')
        pruned_classifier = DecisionTreeClassifier(random_state=0, ccp_alpha=candidate_alpha)
        pruned_classifier.fit(training_set.samples, training_set.labels)
        train_score = pruned_classifier.score(training_set.samples, training_set.labels)
        test_score = pruned_classifier.score(test_set.samples, test_set.labels)

        pruned_classifiers.append(pruned_classifier)
        train_scores.append(train_score)
        test_scores.append(test_score)

        print(f'Finished fitting tree for alpha {candidate_alpha} with {pruned_classifier.tree_.node_count} nodes '
              f'and max depth {pruned_classifier.tree_.max_depth} '
              f'| training score: {train_score} '
              f'| test score: {test_score}')

        # stopping condition: check for certain number of iterations and monotonically decreasing
        if len(test_scores) % 20 == 0 and np.all(np.diff(test_scores[-20:]) <= 0):
            break

    candidate_alphas = candidate_alphas[:len(pruned_classifiers)]

    print('Finished pruning trees')
    print(f'\t Train scores: {train_scores}')
    print(f'\t Train scores: {test_scores}')

    # Visualize number of nodes vs alpha
    node_counts = [classifiers.tree_.node_count for classifiers in pruned_classifiers]
    depth = [classifier.tree_.max_depth for classifier in pruned_classifiers]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(candidate_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("Alpha")
    ax[0].set_ylabel("# of Node")
    ax[0].set_title("Number of Nodes vs Alpha")
    ax[1].plot(candidate_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("Alpha")
    ax[1].set_ylabel("Tree depth")
    ax[1].set_title("Depth vs Alpha")
    #fig.savefig(f'graphs/decision tress/{task.name} - nodes+depth vs alpha.png')

    # Visualize accuracy
    fig, ax = plt.subplots()
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Alpha for Training and Testing Sets")
    ax.plot(candidate_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(candidate_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    #fig.savefig(f'graphs/decision tress/{task.name} - accuracy.png')

    # return best classifier and error
    best_index = np.argmax(test_scores)
    best_classifier = pruned_classifiers[best_index]
    error = 1 - test_scores[best_index]
    print(f'The best classifiers is at index {best_index} with an alpha of {candidate_alphas[best_index]}')

    return best_classifier, error


def shuffle_prune(task: Task, percent_training: float = 0.9, iterations: int = 10, streak: int = 12):
    candidate_classifiers = []
    for iteration in range(iterations):
        print(f'Starting iteration {iteration}')
        training_set, test_set = data_utils.get_training_and_test_sets(task, percent_training, randomize=True)
        base_classifier = DecisionTreeClassifier()

        # Get candidate alpha values for pruning
        path = base_classifier.cost_complexity_pruning_path(training_set.samples, training_set.labels)
        candidate_alphas = np.array(path.ccp_alphas)
        print(f'Found {len(candidate_alphas)} candidate alphas')
        candidate_alphas = candidate_alphas.round(decimals=6)
        candidate_alphas = np.unique(candidate_alphas)
        print(f'Trimmed to {len(candidate_alphas)} candidate alphas')

        # Train pruned classifiers
        pruned_classifiers = []
        train_scores = []
        test_scores = []
        for candidate_alpha in candidate_alphas:
            print(f'Fitting decision tree for alpha {candidate_alpha}...')
            pruned_classifier = DecisionTreeClassifier(random_state=0, ccp_alpha=candidate_alpha)
            pruned_classifier.fit(training_set.samples, training_set.labels)
            train_score = pruned_classifier.score(training_set.samples, training_set.labels)
            test_score = pruned_classifier.score(test_set.samples, test_set.labels)

            pruned_classifiers.append(pruned_classifier)
            train_scores.append(train_score)
            test_scores.append(test_score)

            print(f'Finished fitting tree for alpha {candidate_alpha} with {pruned_classifier.tree_.node_count} nodes '
                  f'and max depth {pruned_classifier.tree_.max_depth} '
                  f'| training score: {train_score} '
                  f'| test score: {test_score}')

            # stopping condition: check for certain number of iterations and monotonically decreasing
            if len(test_scores) % streak == 0 and np.all(np.diff(test_scores[-streak:]) <= 0):
                break

        best_index = np.argmax(test_scores)
        best_classifier = pruned_classifiers[best_index]
        test_error = 1 - test_scores[best_index]
        train_error = 1 - train_scores[best_index]
        candidate_classifiers.append((best_classifier, test_error, train_error))
        print(f'Finished iteration {iteration} with best error {test_error}')

    print(f'Candidate classifiers are {candidate_classifiers}')
    return sorted(candidate_classifiers, key=lambda classifier_error_pair: classifier_error_pair[1])[0]


def run_and_visualize(task: Task, function):
    classifier, error = function(task)
    visualize(classifier, task)


def visualize(classifier: DecisionTreeClassifier, task: Task, function):
    filename = f'graphs/decision trees/{task.name} - {function.__name__}'
    export = tree.export_graphviz(classifier, out_file=None,
                                  feature_names=task.value.features,
                                  class_names=task.value.classes,
                                  filled=True, rounded=True,
                                  special_characters=True,
                                  impurity=True)
    graph = graphviz.Source(export)
    graph.render(filename)


def statistics(clf: DecisionTreeClassifier):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    leaves = []
    stack = [(0, 0)]
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            leaves.append(node_id)

    median_leaf_depth = np.median(node_depth[leaves])

    print(f'Total Nodes: {n_nodes} | Max Depth: {clf.tree_.max_depth} | Leaf Nodes: {clf.tree_.n_leaves} | Median leaf depth: {median_leaf_depth}')
    return n_nodes, clf.tree_.max_depth, clf.tree_.n_leaves, median_leaf_depth


def create_learning_curve(iterations: int = 1, streak: int = 10):
    fig, ax = plt.subplots(1, 2)

    ax[0].set_title("Learning Curve")
    ax[0].set_xlabel("Percentage Training Set")
    ax[0].set_ylabel("Error")

    ax[1].set_title("Training Time")
    ax[1].set_xlabel("Percentage Training Set")
    ax[1].set_ylabel("Training Time (in Seconds)")

    percentages = np.linspace(0, 1, 11)[1:-1]
    for task, name, linestyle in [(Task.SCRIBE_RECOGNITION, 'Scribe', 'dotted'), (Task.LETTER_RECOGNITION, 'Letter', 'dashed')]:
        test_errors = []
        train_errors = []
        training_times = []
        for percent_training in percentages:
            start = time()
            _, test_error, train_error = shuffle_prune(task, percent_training=percent_training, iterations=iterations, streak=streak)
            end = time()

            test_errors.append(test_error)
            train_errors.append(train_error)
            training_times.append(round(end - start, 2))

        ax[0].plot(percentages, test_errors, label=f'{name} Test Error', marker="o", drawstyle="steps-post", linestyle=linestyle)
        ax[0].plot(percentages, train_errors, label=f'{name} Train Error', marker="o", drawstyle="steps-post", linestyle=linestyle)
        ax[1].plot(percentages, training_times, label=f'{name} Classifier', marker="o", drawstyle="steps-post", linestyle=linestyle)

    ax[0].legend(loc="best")
    ax[1].legend(loc="best")
