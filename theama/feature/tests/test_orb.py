"""
Author: David Torpey

License: Apache 2.0

Module with unit tests for the ORB implementation.
"""

import unittest

import numpy as np

from theama.feature import ORB
from theama.utils.utils import load_lena


class OrbTests(unittest.TestCase):
    """
    Class for ORB unit tests.
    """

    def setUp(self):
        self.image = load_lena()
        self.orb = ORB()

    def test_orb_detect_valid_image(self):
        """Function to test successful detection
        of ORB keypoints for a valid image.
        Asserts that number of returned keypoints
        is greater than zero for a known image.
        """

        keypoints = self.orb.detect(self.image)

        self.assertGreater(len(keypoints), 0)

    def test_orb_detect_invalid_image(self):
        """Function to detect successful raising
        of an exception in the case of an invalid
        image being passed to the keypoint detection
        function. Asserts that exception is raised.
        """

        invalid_image = self.image.astype('float32')

        with self.assertRaises(Exception) as context:
            self.orb.detect(invalid_image)

        self.assertTrue('Ensure dtype is uint8.' in str(context.exception))

    def test_orb_describe_valid_image(self):
        """Function to test successful computation
        of ORB descriptors given valid image and
        keypoints. Asserts that at least 1 descriptor
        is returned for a known image.
        """

        keypoints = self.orb.detect(self.image)

        descriptors = self.orb.describe(self.image, keypoints)

        self.assertGreater(len(descriptors), 0)

    def test_orb_describe_invalid_image(self):
        """Function to detect successful raising of
        an exception in the case of an invalid image
        being passed to the descriptor computation
        function. Asserts that exception raised.
        """

        keypoints = self.orb.detect(self.image)
        invalid_image = self.image.astype('float32')

        with self.assertRaises(Exception) as context:
            self.orb.describe(invalid_image, keypoints)

        self.assertTrue('Ensure dtype is uint8.' in str(context.exception))

    def test_orb_detect_invalid_image_channels(self):
        """Function to detect successful raising of
        an exception in the case of an invalid image
        being passed to the detection computation
        function. Asserts that exception raised.
        """

        invalid_image = np.zeros((32, 32, 4), dtype='uint8')

        with self.assertRaises(Exception) as context:
            self.orb.detect(invalid_image)

        self.assertTrue('Must be an image with 1 or 3 channels.' in str(context.exception))

    def test_orb_describe_invalid_keypoints(self):
        """Function to detect successful raising of
        an exception in the case of invalid keypoints
        being passed to the descriptor computation
        function. Asserts that exception raised.
        """

        invalid_keypoints = [1, 2, 3]

        with self.assertRaises(Exception) as context:
            self.orb.describe(self.image, invalid_keypoints)

        self.assertTrue('Please input keypoints of type cv2.KeyPoint.' in str(context.exception))

    def test_orb_keypoints_exist(self):
        """Function to detect successful raising of
        an exception in the case of no keypoints
        being passed to the descriptor computation
        function. Asserts that exception raised.
        """

        keypoints = []

        with self.assertRaises(Exception) as context:
            self.orb.describe(self.image, keypoints)

        self.assertTrue('No keypoints.' in str(context.exception))
