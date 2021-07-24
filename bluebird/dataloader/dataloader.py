"""
DataLoader
==========

Simplest of the dataloader classes.
Uses the data that has been already loaded into memory.
"""

from .dataloaderbase import DataLoaderBase

from bluebird.data import Batch
from bluebird.tensor import Tensor

class DataLoader(DataLoaderBase):
    def __init__(self, inputs: Tensor, targets: Tensor, batch_size: int = 32, shuffle: bool = True):
        """
        Initalizes the object.

        Args:
            inputs (:obj:`Tensor`): inputs (data that passes throughout the network)
            targets (:obj:`Tensor`): target data
            batch_size (int): length of every batch size, defaults to 32
            shuffle (bool): shuffles data if true, defaults to True

        """
        super().__init__(batch_size, shuffle)

        self.inputs = inputs
        self.targets = targets


    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Batch:
        end = idx + self.batch_size
        batch_inputs = self.inputs[idx:end]
        batch_targets = self.targets[idx:end]

        return Batch(batch_inputs, batch_targets)

