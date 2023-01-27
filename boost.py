import data_utils
from data_utils import Task
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import ClassifierMixin


### Docs used:
# https://scikit-learn.org/stable/modules/ensemble.html#adaboost
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-multiclass-py

def learn(task: Task, weak_classifier: ClassifierMixin = None):
    training_set, test_set = data_utils.get_training_and_test_sets(task)
    classifier = AdaBoostClassifier(
        n_estimators=100,
        estimator=weak_classifier,
        algorithm='SAMME',
        learning_rate=1.0
    )

    classifier.fit(training_set.samples, training_set.labels)

    error = 1.0 - classifier.score(test_set.samples, test_set.labels)

    return classifier, error
