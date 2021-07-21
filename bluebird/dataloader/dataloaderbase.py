"""
DataLoader
==========

Data loader is a special class used for loading training data.

It can also be used to load data direcly from folders or to create a custom loader to fit your every need.
"""

import numpy as np

from bluebird.data import DataIterator, Batch
from bluebird.tensor import Tensor
from typing import Iterator

class DataLoaderBase(DataIterator):
    """
    Base class that every data loader inherits.

    Example::

        class CustomDataLoader(DataLoaderBase):
            def __len__(self) -> int:
                return len(x_train)

            def __getitem__(self, idx) -> int:
                x = x_train[idx: idx+self.batch_size]
                y = y_train[idx: idx+self.batch_size]

                return Batch(x, y)
    """

    def __init__(self, batch_size = 32, shuffle = True):
        """
        Initalizes the object.

        Args:
            batch_size (int): length of every batch size, defaults to 32
            shuffle (bool): shuffles data if true, defaults to True

        """
        super().__init__()

        if not isinstance(batch_size, int):
            raise TypeException("batch_size", "int")

        if not isinstance(shuffle, int):
            raise TypeException("shuffle", "int")

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        """
        Used to get length of dataset.

        returns:
            int: length of dataset
        """

        raise NotImplementedError

    def __getitem__(self, idx) -> Batch:
        """
        Retursn a batch.

        Args:
            idx (int): start index of batch

        Returns:
            Batch: batch of data with inputs and targets

        """

        raise NotImplementedError

    def __call__(self) -> Iterator[Batch]:
        """
        Returns batches of data.

        Returns: 
            Iterator[Batch]: Batches of data

        """
        idxes = np.arange(0, self.__len__(), self.batch_size)

        if self.shuffle:
            np.random.shuffle(idxes)

        for idx in idxes:
            yield self.__getitem__(idx)
