from typing import Tuple

from enum import Enum
import numpy


LETTER_LABELS = ['Box X Position', 'Box Y Position', 'Box Width', 'Box Height', 'Total Pixels',
                 'Pixel Mean X-Coor.', 'Pixel Mean Y-Coor', 'X Variance', 'Y Variance', 'Mean XY Correlation',
                 'Mean of X*X*Y', 'Mean of X*Y*Y', 'X Edge Count Mean', 'xegvy', 'Y Edge Count Mean', 'yegvx']
LETTER_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                  'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


class Task(Enum):
    LETTER_RECOGNITION = ('letter recognition', 0, numpy.arange(1, 17, dtype=int), int)  # directory, index of label, data columns, data type
    SCRIBE_RECOGNITION = ('scribe recognition', 10, numpy.arange(0, 10, dtype=int), float)


class SampleSet:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self._label_int_map = None

    def size(self) -> int:
        return self.labels.size

    def num_features(self) -> int:
        return self.samples.shape[1]

    def num_classes(self) -> int:
        return len(numpy.unique(self.labels))

    def labels_as_ints(self):
        label_map = self._label_int_map or self._create_labels_to_int_map()
        map_function = numpy.vectorize(lambda label: label_map[label])
        return map_function(self.labels)

    def use_label_to_int_map_from(self, other: 'SampleSet'):
        self._label_int_map = other._label_int_map

    def _create_labels_to_int_map(self):
        self._label_int_map = {}
        for index, label in enumerate(numpy.unique(self.labels)):
            self._label_int_map[label] = index
        return self._label_int_map


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

