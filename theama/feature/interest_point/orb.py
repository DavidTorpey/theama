"""
Author: David Torpey

License: Apache 2.0

Module for ORB interest points detection and
descriptor computation.
"""

import cv2


class ORB(object):
    """
    Class to facilitate the detection and description
    of ORB interest points and resultant descriptors.
    """

    def __init__(self):
        self.orb = cv2.ORB_create()

    def detect(self, input_image):
        """Function to detect ORB interest points.

        Args:
            input_image: Input image. Must be of type uint8.

        Returns:
            List of KeyPoint objects defining the ORB
            interest points.
        """

        if input_image.dtype.name != 'uint8':
            raise Exception('Ensure dtype is uint8.')

        if len(input_image.shape) > 3 or \
                len(input_image.shape) < 2 or \
                (len(input_image.shape) == 3 and
                 input_image.shape[-1] not in [1, 3]):
            raise Exception('Must be an image with 1 or 3 channels.')

        return self.orb.detect(input_image)

    def describe(self, input_image, keypoints):
        """Function to compute the ORB descriptors
        for a list of given ORB interest points.

        Args:
            input_image: Input image. Must be of type uint8.
            keypoints: List of KeyPoint objects defining
                       ORB interest points.

        Returns:
            NumPy array of ORB descriptors.
        """

        if input_image.dtype.name != 'uint8':
            raise Exception('Ensure dtype is uint8.')

        if len(input_image.shape) > 3 or \
                len(input_image.shape) < 2 or \
                (len(input_image.shape) == 3 and
                 input_image.shape[-1] not in [1, 3]):
            raise Exception('Must be an image with 1 or 3 channels.')

        if not keypoints:
            raise Exception('No keypoints.')

        if not isinstance(keypoints[0], cv2.KeyPoint):
            raise Exception('Please input keypoints of type cv2.KeyPoint.')

        _, descriptors = self.orb.compute(input_image, keypoints)

        return descriptors
