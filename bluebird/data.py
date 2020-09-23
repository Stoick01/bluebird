"""
Feed inputs in batches
"""

from typing import Iterator, NamedTuple

import numpy as np

from .tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])

class DataIterator:
    """
    Base class that every data iterator inherits

    Example:
        class CustomDataIterator(DataIterator):
            def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
                ... Make sure to return a list of batches (Named tuple with inputs and targets)
    """

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError


class BatchIterator(DataIterator):
    """
    Default iterator, returns batches of input, target pairs

    Args:
        batch_size: length of every batch size
            default: 32
            type: int
        shuffle: shuffles data if true
            default: True
            type: bool

    Example:
        
        >>> iterator = BatchIterator()
        >>> for batch in iterator(Data):
                ....

    """

    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        if not isinstance(batch_size, int):
            raise TypeException("batch_size", "int")

        if not isinstance(shuffle, int):
            raise TypeException("shuffle", "int")

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        """
        Returns batches of data

        Args:
            inputs: Type: Tensor
            targets: Type: Tensor

        Returns: Iterable Batches(Named tuple, with targets and inputs)

        Example:
            
            >>> iterator = BatchIterator()
            >>> for batch in iterator(Data):
                    ....

        """

        starts = np.arange(0, len(inputs), self.batch_size)

        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]

            yield Batch(batch_inputs, batch_targets)