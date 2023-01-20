from typing import Tuple

from enum import Enum
import numpy


class Task(Enum):
    LETTER_RECOGNITION = ('letter recognition', 0, numpy.arange(1, 17, dtype=int), int)  # directory, index of label, data columns, data type
    SCRIBE_RECOGNITION = ('scribe recognition', 10, numpy.arange(0, 10, dtype=int), float)


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
