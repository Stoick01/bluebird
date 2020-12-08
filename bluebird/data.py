"""
Data Iterators
==============

Designed to feed the network data for training.

Iterators return Batch datatype, which is named tuple with inputs and targets tensors:

``Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])``
"""

from typing import Iterator, NamedTuple

import numpy as np

from .tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])

class DataIterator:
    """
    Base class that every data iterator inherits.

    Example::

        class CustomDataIterator(DataIterator):
            def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
                for inp, targ in zip(inputs, targets):
                    yield Batch(inp, targ)
    """

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError


class BatchIterator(DataIterator):
    """
    Default iterator, returns batches of input, target pairs.

    Batches are randomized by default.

    Example::

        iterator = BatchIterator()
        for batch in iterator(Data):
            # do something

    :obj:`__call__(inputs: Tensor, targets: Tensor) -> Iterator[Batch]:`

    Returns batches of data.

        Args:
            inputs (:obj:`Tensor`): input to the network
            targets (:obj:`Tensor`): expected value

        Returns: 
            Iterator[Batch]: Batches of data
    
    """

    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        """
        Initalizes the object.

        Args:
            batch_size (int): length of every batch size, defaults to 32
            shuffle (bool): shuffles data if true, defaults to True

        """

        if not isinstance(batch_size, int):
            raise TypeException("batch_size", "int")

        if not isinstance(shuffle, int):
            raise TypeException("shuffle", "int")

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        """
        Returns batches of data.

        Args:
            inputs (:obj:`Tensor`): input to the network
            targets (:obj:`Tensor`): expected value

        Returns: 
            Iterator[Batch]: Batches of data

        """

        starts = np.arange(0, len(inputs), self.batch_size)

        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]

            yield Batch(batch_inputs, batch_targets)