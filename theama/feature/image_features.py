"""
Author: David Torpey

License: Apache 2.0

Redistribution Licences:
- Scikit-Image: https://scikit-image.org/docs/dev/license.html

Module for image feature computation.
"""

from skimage.feature import hog, local_binary_pattern


def compute_hog(input_image, *args, **kwargs):
    """

    Args:
        input_image:
        *args:
        **kwargs:

    Returns:

    """

    return hog(
        input_image,
        *args,
        **kwargs
    )


def compute_lbp(input_image, radius=3, n_points=24, *args, **kwargs):
    """

    Args:
        input_image:
        radius:
        n_points:
        *args:
        **kwargs:

    Returns:

    """

    return local_binary_pattern(input_image, radius, n_points, *args, **kwargs)
