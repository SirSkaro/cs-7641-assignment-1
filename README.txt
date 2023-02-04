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

# Decision Tree
Import the modules data_utils and decision_tree. With both imported, you can call any of the methods in the decision_tree module. Most methods to create a classifier
only have 1 required argument - the Task enum. Other utility methods for printing statistics and visualization also exist.

To recreate the learning curve, simply call decision_tree.create_learning_curve. The parameters used for the graph in the paper were "iterations: int = 5, streak: int = 10".

# KNN