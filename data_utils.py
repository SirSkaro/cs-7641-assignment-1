from typing import Tuple

from enum import Enum
import numpy


class Task(Enum):
    LETTER_RECOGNITION = ('letter recognition', 0, numpy.arange(1, 17, dtype=int), int)  # directory, index of label, data columns, data type
    SCRIBE_RECOGNITION = ('scribe recognition', 10, numpy.arange(0, 10, dtype=int), float)


class SampleSet:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def size(self) -> int:
        return self.labels.size


def parse_data(task: Task) -> Tuple[numpy.ndarray, numpy.array]:
    filename = './datasets/' + task.value[0] + '/data'
    dataset = numpy.loadtxt(
        fname=filename,
        delimiter=',',
        usecols=task.value[2],
        dtype=task.value[3]
    )
    labels = numpy.loadtxt(
        fname=filename,
        delimiter=',',
        usecols=task.value[1],
        dtype=str
    )

    return dataset, labels


def partition_samples(samples: numpy.ndarray, labels: numpy.array):
    sample_count = labels.size
    training_set_size = int(sample_count * 0.9)

    training_set = SampleSet(samples[:training_set_size], labels[:training_set_size])
    test_set = SampleSet(samples[training_set_size:], labels[training_set_size:])
    return training_set, test_set


def get_training_and_test_sets(task: Task) -> Tuple[SampleSet, SampleSet]:
    samples, labels = parse_data(task)
    return partition_samples(samples, labels)

