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
The different datasets are reference via the Task enum. To manually get a training set and a test set, simply use data_utils.get_training_and_test_sets and pass in the desired
Task enum. There are optional parameters for partition percentage and shuffling.

# Graphs
Graphs included in the write up and located in the "graphs" directory in the top-level of the project

# Training the classifiers
The below sections explain how to run the code for each classifier. There are concrete examples in the file "scratch code.txt".

## Decision Tree
Import the modules data_utils and decision_tree. With both imported, you can call any of the methods in the decision_tree module. Most methods to create a classifier
only have 1 required argument - the Task enum. Other utility methods for printing statistics and visualization also exist.

To recreate the learning curve, simply call decision_tree.create_learning_curve. The parameters used for the graph in the paper were "iterations: int = 5, streak: int = 10".

Example:
"""
import data_utils
from data_utils import Task
import decision_tree

# Create classifiers from the initial configuration
letter_classifier, letter_error = decision_tree.learn(Task.LETTER_RECOGNITION)
scribe_classifier, scribe_error = decision_tree.learn(Task.SCRIBE_RECOGNITION)

# Create classifiers from the shuffle + prune configuration
letter_classifier, letter_error = decision_tree.shuffle_prune(Task.LETTER_RECOGNITION, iterations = 5)
scribe_classifier, scribe_error = decision_tree.shuffle_prune(Task.SCRIBE_RECOGNITION, iterations = 5)
"""


## KNN
Import the modules data_utils and decision_tree. With both imported, you can call any of the methods in the decision_tree module. Most methods to create a classifier
only have 1 required argument - the Task enum. Other utility methods for printing statistics and visualization also exist.

To recreate the learning curve, simply call decision_tree.create_learning_curve. The parameters used for the graph in the paper were the default parameters.

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


