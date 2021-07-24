"""
FileLoader
==========

Type of data loader class that is used to directly load data from directory.

Loader uses pillow for loading images and converts them into tensor.

It requires specific structure of directory:
    ├── dataset                     
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ... 
    └── ....
"""

import os

from PIL import Image
import numpy as np

from .dataloaderbase import DataLoaderBase

from bluebird.data import Batch
from bluebird.tensor import Tensor

class ImageLoader(DataLoaderBase):
    def __init__(self, image_dir: str, targets: Tensor, channels: int = 3, im_shape: tuple = None, batch_size: int = 32, shuffle: bool = True):
        """
        Initalizes the object.

        Args:
            image_dir (str): location of base images directory
            targets (:obj:`Tensor`): target data, note: order must be the same as filenames in directory
            channels (int): number of channels of images, 1: grayscale, 3: rgb, 4: rgba, defautls to 3
            im_shape (tuple, optional): define if you want to reshape the image, None = default image shape
            batch_size (int, optional): length of every batch size, defaults to 32
            shuffle (bool, optional): shuffles data if true, defaults to True

        """
        super().__init__(batch_size, shuffle)

        self.image_dir = image_dir
        self.targets = targets

        self.im_shape = im_shape

        if channels == 1:
            self.mode = 'L'
        elif channels == 4:
            self.mode = 'RGBA'
        else:
            self.mode = 'RGB'

        self.preload()

    def preload(self) -> None:
        """
        Loads the file names as Tensor        
        """

        self.names = os.listdir(self.image_dir)


    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx) -> Batch:
        end = idx + self.batch_size
        images = self.names[idx:end]
        batch_targets = self.targets[idx:end]

        batch_inputs = []

        for im in images:
            i = Image.open(os.path.join(self.image_dir, im))
            i = i.convert(self.mode)

            if self.im_shape != None:
                i = i.resize(self.im_shape, Image.ANTIALIAS)

            i = np.array(i)

            batch_inputs.append(i)

        
        batch_inputs = np.array(batch_inputs)

        return Batch(batch_inputs, batch_targets)