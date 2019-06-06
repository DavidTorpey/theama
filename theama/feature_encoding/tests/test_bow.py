"""
Author: David Torpey

License: Apache 2.0

Module with unit tests for the BoW implementation.
"""

import unittest

import numpy as np

from theama.feature_encoding import BOW


class BoWTests(unittest.TestCase):
    """
    Class for BoW unit tests.
    """

    def setUp(self):
        self.codebook_size = 32
        self.bow = BOW(self.codebook_size)
        self.feature_dimensionality = 128
        self.dummy_descriptors = \
            np.random.random((100, self.feature_dimensionality))

    def tearDown(self):
        pass

    def test_codebook_creation(self):
        """Function to test successful learning of the BoW
        codebook. Asserts that codebook is initialised.
        """

        self.bow.learn_codebook(self.dummy_descriptors)

        self.assertTrue(self.bow.codebook is not None)

    def test_compute_bow_without_codebook_creation(self):
        """Function to test successful error raising when
        compute_feature is called without a prior
        call to the learn_codebook method. Asserts that
        exception is raised.
        """

        with self.assertRaises(Exception) as context:
            self.bow.compute_feature_vector(self.dummy_descriptors)

        self.assertTrue('Please run learn_codebook method.' in str(context.exception))

    def test_compute_bow_with_codebook_creation(self):
        """Function to test BoW descriptors are successfully
        computed when a prior call to learn_codebook has been
        made. Asserts that the resulting BoW descriptor
        dimensionality is correct.
        """

        self.bow.learn_codebook(self.dummy_descriptors)

        bow_descriptor = self.bow.compute_feature_vector(self.dummy_descriptors)
        bow_descriptor_dimension = len(bow_descriptor)

        self.assertEqual(
            bow_descriptor_dimension,
            self.codebook_size
        )
