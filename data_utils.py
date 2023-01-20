from typing import Tuple

from enum import Enum
import numpy


class DataSet(Enum):
    LETTER_RECOGNITION = ('letter recognition', 0, numpy.arange(1, 17, dtype=int), int)  # directory, index of label, data columns, data type
    SCRIBE_RECOGNITION = ('scribe recognition', 10, numpy.arange(0, 10, dtype=int), float)


def parse_data(dataset: DataSet) -> Tuple[numpy.ndarray, numpy.array]:
    filename = './datasets/' + dataset.value[0] + '/data'
    data = numpy.loadtxt(
        fname=filename,
        delimiter=',',
        usecols=dataset.value[2],
        dtype=dataset.value[3]
    )
    labels = numpy.loadtxt(
        fname=filename,
        delimiter=',',
        usecols=dataset.value[1],
        dtype=str
    )

    return data, labels
