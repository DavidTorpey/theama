"""
Author: David Torpey

License: Apache 2.0

Redistribution Licensing:
- OpenCV: https://opencv.org/license/

Module for BRISK interest points detection and
descriptor computation.
"""

import cv2


class BRISK(object):
    """
    Class to facilitate the detection and description
    of BRISK interest points and resultant descriptors.
    """

    def __init__(self):
        self.brisk = cv2.BRISK_create()

    def detect(self, input_image):
        """Function to detect BRISK interest points.

        Args:
            input_image: Input image. Must be of type uint8.

        Returns:
            List of KeyPoint objects defining the BRISK
            interest points.
        """

        assert input_image.dtype.name == 'uint8'

        return self.brisk.detect(input_image)

    def describe(self, input_image, keypoints):
        """Function to compute the BRISK descriptors
        for a list of given BRISK interest points.

        Args:
            input_image: Input image. Must be of type uint8.
            keypoints: List of KeyPoint objects defining
                       BRISK interest points.

        Returns:
            NumPy array of BRISK descriptors.
        """

        assert input_image.dtype.name == 'uint8'

        _, descriptors = self.brisk.compute(input_image, keypoints)

        return descriptors
