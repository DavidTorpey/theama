"""
Author: David Torpey

License: Apache 2.0

Redistribution Licensing:
- NumPy: https://www.numpy.org/license.html#

Module with unit tests for the VLAD implementation.
"""

import unittest

import numpy as np

from theama.feature_encoding import VLAD


class VLADTests(unittest.TestCase):
    """
    Class for VLAD unit tests.
    """

    def setUp(self):
        self.K = 32
        self.vlad = VLAD(self.K)
        self.D = 128
        self.dummy_descriptors = np.random.random((100, self.D))

    def tearDown(self):
        pass

    def test_codebook_creation(self):
        """Function to test successful learning of the VLAD
        codebook. Asserts that codebook is initialised.
        """

        self.vlad.learn_codebook(self.dummy_descriptors)

        self.assertTrue(self.vlad.codebook is not None)

    def test_compute_vlad_without_codebook_creation(self):
        """Function to test successful error raising when
        compute_vlad_descriptor is called without a prior
        call to the learn_codebook method. Asserts that
        exception is raised.
        """

        with self.assertRaises(Exception) as context:
            self.vlad.compute_feature_vector(self.dummy_descriptors)

        self.assertTrue('Please run learn_codebook method.' in str(context.exception))

    def test_compute_vlad_with_codebook_creation(self):
        """Function to test VLAD descriptors are successfully
        computed when a prior call to learn_codebook has been
        made. Asserts that the resulting VLAD descriptor
        dimensionality is correct.
        """

        self.vlad.learn_codebook(self.dummy_descriptors)

        vlad_descriptor = self.vlad.compute_feature_vector(self.dummy_descriptors)
        vlad_descriptor_dimension = len(vlad_descriptor)

        self.assertEqual(vlad_descriptor_dimension, self.D * self.K)