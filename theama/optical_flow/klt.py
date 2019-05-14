"""
Author: Ziyad Jappie

License: Apache 2.0

Redistribution Licensing:
- OpenCV: https://opencv.org/license/
- NumPy: https://www.numpy.org/license.html#

Implementation adapted from: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

Module containing a Python implementation of the Lucas-Kanade
optical flow algorithm using the Shi-Thomasi algorithm
to detect features.
"""

import numpy as np
import cv2


class Lucas_Kanade_OF(object):
    """
    Class for the implementation of the
    Lucas-Kanade optical flow algorithm.
    """

    def __init__(self, feature_params=None, lk_params=None):
        self.feature_params = feature_params
        self.lk_params = lk_params

    def perform_optical_flow(self, video, recompute_lost_points=True):
        """Function to compute optical flow by using
        the Lucas-Kanade algorithm. This is an example of sparse optical flow
        which computes the optical flow for selected points in a given frame.

        Returns the set of points tracked throughout the video

        Args:
            video: is the video fed in as a numpy array.
            recompute_lost_points: If 'True', once tracked points are lost
            new features are computed to be tracked. If 'False' only original
            points are tracked and returned. Default is 'True'.
        """

        if type(video) is not np.ndarray:
            raise Exception("Not a numpy array")
        if len(video.shape) < 3 or len(video.shape) > 4:
            raise Exception("Not a video numpy file")

        # params for ShiTomasi corner detection
        if self.feature_params is None:
            self.feature_params = dict(maxCorners=100,
                                       qualityLevel=0.3,
                                       minDistance=7,
                                       blockSize=7)

        # Parameters for lucas kanade optical flow
        if self.lk_params is None:
            self.lk_params = dict(winSize=(15, 15),
                                  maxLevel=2,
                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        video = video.astype('uint8')
        f, r, c, d = video.shape
        old_gray = cv2.cvtColor(video[0], cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)

        points = []
        for i in range(1, f):
            frame_gray = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)
            try:
                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]

            except TypeError:
                print "Original points lost at frame number ", i, "of ", f
                if recompute_lost_points:
                    print "Computing new features to track"
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)
                    good_new = p1[st == 1]
                else:
                    break

            good_new = np.array(good_new)
            points.append(good_new)

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        return np.array(points)
