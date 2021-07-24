import unittest

import os
import shutil
import numpy as np
from PIL import Image

from bluebird.dataloader import ImageLoader


class TestImageLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.exists('temp'):
            os.mkdir('temp')

        for i in range(128):
            im = np.random.randn(32, 32, 3) * 255
            im = Image.fromarray(im.astype('uint8')).convert('RGB')
            im.save('temp/%000d.jpg' % i)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('temp')

    def test_default(self):
        """Tests ImageLoader with default arguments"""
        loader = ImageLoader('temp', np.random.randn(128, 1), batch_size=16)
        for batch in loader():
            self.assertEqual(batch.inputs.shape, (16, 32, 32, 3))
            self.assertEqual(batch.targets.shape, (16, 1))

    def test_grayscale(self):
        """Tests ImageLoader when image is converted to grayscale"""
        loader = ImageLoader('temp', np.random.randn(128, 1), batch_size=16, channels=1)
        for batch in loader():
            self.assertEqual(batch.inputs.shape, (16, 32, 32))
            self.assertEqual(batch.targets.shape, (16, 1))

    def test_rgba(self):
        """Tests ImageLoader when image is converted to rgba"""
        loader = ImageLoader('temp', np.random.randn(128, 1), batch_size=16, channels=4)
        for batch in loader():
            self.assertEqual(batch.inputs.shape, (16, 32, 32, 4))
            self.assertEqual(batch.targets.shape, (16, 1))

    def test_resize(self):
        """Tests ImageLoader when image is resized"""
        loader = ImageLoader('temp', np.random.randn(128, 1), batch_size=16, im_shape=(16, 16))
        for batch in loader():
            self.assertEqual(batch.inputs.shape, (16, 16, 16, 3))
            self.assertEqual(batch.targets.shape, (16, 1))