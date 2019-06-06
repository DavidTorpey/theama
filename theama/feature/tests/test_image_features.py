"""
Author: David Torpey

License: Apache 2.0

Redistribution Licensing:
- NumPy: https://www.numpy.org/license.html#

Module with unit tests for the image features.
"""

import unittest

import numpy as np

from theama.feature import compute_hog
from theama.utils.utils import load_lena


class ImageFeatureTest(unittest.TestCase):

    def setUp(self):
        self.image = load_lena()

    def test_hog_valid_image(self):
        hog_vector = compute_hog(self.image)

        self.assertGreater(len(hog_vector), 0)

    def test_hog_invalid_image(self):
        invalid_image = np.zeros((1, 1))

        with self.assertRaises(ValueError) as context:
            compute_hog(invalid_image)

        self.assertTrue('negative dimensions are not allowed' in str(context.exception))
