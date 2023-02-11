# Project Setup
* Setup and activate a virtual environment with Python 3.10.
    * If using Anaconda/miniconda, you can simply use a command like "conda create -n bchurchill6_assignment1 python=3.10".
    * Otherwise, download a distribution of Python 3.10 manually and use it to run "python -m venv bchurchill6_assignment1"
* Install requirements. With the base of the project as the working directory, you can run the command "pip install -r requirements.txt"

# The Data
All data is located in /datasets. Each dataset is located its own directory with the filename "data" along with a description of the data (description.txt) and a link to the original download location (UCI).

!!! IMPORTANT !!!
Interaction in the code with the data has all be abstracted into the data_utils module. It includes utils to create objects that represent sample sets and
partition the dataset. It is required that you import it to interact with any of the functions defined in the classifier modules.
The different datasets are reference via the Task enum. To manually get a training set and a test set, simply use data_utils.get_training_and_test_sets() and pass in the desired
Task enum. There are optional parameters for partition percentage and shuffling.

# Graphs
Graphs included in the write up and located in the "graphs" directory in the top-level of the project

# Training the classifiers
The below sections explain how to run the code for each classifier. There are concrete examples in the file "scratch code.txt".


## Decision Tree
Import the modules data_utils and decision_tree. With both imported, you can call any of the methods in the decision_tree module. Most methods to create a classifier
only have 1 required argument - the Task enum. Other utility methods for printing statistics and visualization also exist.

To recreate the learning curve, simply call decision_tree.create_learning_curve(). The parameters used for the graph in the paper were "iterations: int = 5, streak: int = 10".

Example:
"""
import data_utils
from data_utils import Task
import decision_tree

# Create classifiers from the initial configuration
letter_classifier, letter_error = decision_tree.learn(Task.LETTER_RECOGNITION)
decision_tree.statistics(letter_classifier)
decision_tree.visualize(letter_classifier, Task.LETTER_RECOGNITION)

scribe_classifier, scribe_error = decision_tree.learn(Task.SCRIBE_RECOGNITION)
decision_tree.statistics(scribe_classifier)
decision_tree.visualize(scribe_classifier, Task.SCRIBE_RECOGNITION)

# Create classifiers from the shuffle + prune configuration
letter_classifier, letter_error = decision_tree.shuffle_prune(Task.LETTER_RECOGNITION, iterations = 5)
decision_tree.statistics(letter_classifier)
decision_tree.visualize(letter_classifier, Task.LETTER_RECOGNITION)

scribe_classifier, scribe_error = decision_tree.shuffle_prune(Task.SCRIBE_RECOGNITION, iterations = 5)
decision_tree.statistics(scribe_classifier)
decision_tree.visualize(scribe_classifier, Task.SCRIBE_RECOGNITION)
"""


## KNN
Import the modules data_utils and knn. With both imported, you can call any of the methods in the knn module. Most methods to create a classifier
only have 1 required argument - the Task enum.

To recreate the learning curve, simply call knn.create_learning_curve(). The parameters used for the graph in the paper are demonstrated below.

Example:
"""
import data_utils
from data_utils import Task
import knn


# Create classifiers from the initial configuration
letter_classifier, letter_error = knn.learn(Task.LETTER_RECOGNITION)
scribe_classifier, scribe_error = knn.learn(Task.SCRIBE_RECOGNITION)

# Create classifiers from the Expectation-Maximization Configuration
letter_classifier, letter_error = knn.best_expectation_maximization_shuffle(Task.LETTER_RECOGNITION, iterations=5)
scribe_classifier, scribe_error = knn.best_expectation_maximization_shuffle(Task.SCRIBE_RECOGNITION, iterations=5)
"""


## Boosting
Import the modules data_utils and boost. With both imported, you can call any of the methods in the boost module. The method to create a classifier
only have 2 required arguments - the Task enum and the weak learner template. Other utility methods for printing statistics.

To recreate the learning curve, simply call boost.create_learning_curve(). The parameters used for the graph in the paper are demonstrated below.

Example:
"""
import data_utils
from data_utils import Task
import boost
from sklearn.tree import DecisionTreeClassifier

# Create classifiers from the initial configuration
letter_weak_classifier = DecisionTreeClassifier(max_depth=16)
letter_classifier, letter_error = boost.learn(Task.LETTER_RECOGNITION, weak_classifier=letter_weak_classifier)
boost.statistics(letter_classifier)

scribe_letter_weak_classifier = DecisionTreeClassifier(max_depth=16)
scribe_classifier, scribe_error = boost.learn(Task.SCRIBE_RECOGNITION, weak_classifier=scribe_weak_classifier)
boost.statistics(scribe_classifier)

# Create classifiers from the Shffule + Prune Configuration
letter_weak_classifier = DecisionTreeClassifier(max_depth=20, ccp_alpha=0.0000005)
letter_classifier, letter_error = boost.learn(Task.LETTER_RECOGNITION, weak_classifier=letter_weak_classifier, ensemble_size=100, shuffle=True)
boost.statistics(letter_classifier)

scribe_weak_classifier = DecisionTreeClassifier(max_depth=20, ccp_alpha=0.000005)
scribe_classifier, scribe_error = boost.learn(Task.SCRIBE_RECOGNITION, weak_classifier=scribe_weak_classifier, ensemble_size=100, shuffle=True)
boost.statistics(scribe_classifier)
"""


## SVM
Import the modules data_utils and svm. With both imported, you can call any of the methods in the svm module. Most methods to create a classifier
only have 1 required argument - the Task enum.

To recreate the learning curve, simply call svm.create_learning_curve(). The parameters used for the graph in the paper are demonstrated below.

Example:
"""
import data_utils
from data_utils import Task
import svm

# Create classifiers from the initial configuration
letter_classifier, letter_error = svm.learn(Task.LETTER_RECOGNITION)
scribe_classifier, scribe_error = svm.learn(Task.SCRIBE_RECOGNITION)

# Create classifiers from the Weighted + Shuffle Configuration
letter_classifier, letter_error = svm.iterate_weighted_shuffle(Task.LETTER_RECOGNITION)
scribe_classifier, scribe_error = svm.iterate_weighted_shuffle(Task.SCRIBE_RECOGNITION)
"""


## Neural Networks
Import the modules data_utils and neural_network. With both imported, you can call any of the methods in the neural_network module. Most methods to create a classifier
only have 1 required argument - the Task enum.

To recreate the learning curve, simply call neural_network.create_learning_curve(). The parameters used for the graph in the paper are demonstrated below.

Example:
"""
import data_utils
from data_utils import Task
import neural_network
from neural_network import Activation, Optimizer

# Create classifiers from the initial configuration
letter_classifier, letter_error = neural_network.learn(Task.LETTER_RECOGNITION)
scribe_classifier, scribe_error = neural_network.learn(Task.SCRIBE_RECOGNITION)

# Create classifiers from the Bigger Architecture + Stochastic Gradient Descent Configuration
letter_classifier, letter_error, letter_validation_error = neural_network.learn(Task.LETTER_RECOGNITION, shuffle=True,
                                               hidden_layers=6, optimizer=Optimizer.ADA_DELTA,
                                               activation=Activation.SCALED_EXPONENTIAL_LINEAR_UNIT)

scribe_classifier, scribe_error, scribe_validation_error = neural_network.learn(Task.SCRIBE_RECOGNITION, shuffle=True,
                                               hidden_layers=6, optimizer=Optimizer.ADA_DELTA,
                                               activation=Activation.SCALED_EXPONENTIAL_LINEAR_UNIT)
"""