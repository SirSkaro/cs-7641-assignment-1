import data_utils
from data_utils import Task

import tensorflow as tf
import numpy as np


### Docs used:
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# https://www.tensorflow.org/tutorials/quickstart/beginner
# https://www.tensorflow.org/tutorials/keras/classification


def learn(task: Task):
    training_set, test_set = data_utils.get_training_and_test_sets(task)

    classifier = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(training_set.num_features(),), name='input'),
        tf.keras.layers.Dense(units=4, activation='sigmoid', name='hidden1'),
        tf.keras.layers.Dense(units=training_set.num_classes(), name='output'),
    ])

    classifier.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    converged = False
    required_mean_improvement = 0.015
    epoch = 0
    epochs_per_run = 25
    while not converged:
        history = classifier.fit(training_set.samples, training_set.labels_as_ints(),
                                 epochs=epoch + epochs_per_run, initial_epoch=epoch,
                                 validation_split=0.05, shuffle=True)
        epoch += epochs_per_run
        epoch_validation_accuracy = np.array(history.history['val_accuracy'][-epochs_per_run:])
        improvement = epoch_validation_accuracy.max() - epoch_validation_accuracy.min()
        converged = improvement <= required_mean_improvement
        print('improvement for epoch is ~' + str(improvement))

    test_set.use_label_to_int_map_from(training_set)
    loss, accuracy = classifier.evaluate(test_set.samples, test_set.labels_as_ints(), verbose=2)

    return classifier, accuracy
