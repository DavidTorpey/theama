"""
Author: David Torpey

License: Apache 2.0

General utilities file for theama.
"""

import os

import cv2


def load_lena():
    """Function to read lena image from resources
    directory.

    Returns:
        Loaded lena image as uint8 NumPy array.
    """

    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')
    lena_path = os.path.join(dir_path, 'lena.jpeg')
    return cv2.imread(lena_path)
