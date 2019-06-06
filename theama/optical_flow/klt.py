"""
Author: Ziyad Jappie

License: Apache 2.0

Implementation adapted from:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

Module containing a Python implementation of the Lucas-Kanade
optical flow algorithm using the Shi-Thomasi algorithm
to detect features.
"""

import numpy as np
import cv2


class LucasKanade(object):
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

        if not isinstance(video, np.ndarray):
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
                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                            10, 0.03))

        video = video.astype('uint8')
        frames, _, _, _ = video.shape
        old_gray = cv2.cvtColor(video[0], cv2.COLOR_BGR2GRAY)
        init_points = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)

        points = []
        for i in range(1, frames):
            frame_gray = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            new_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                           frame_gray,
                                                           init_points,
                                                           None,
                                                           **self.lk_params)
            try:
                # Select good points
                good_points_only = new_points[st == 1]

            except TypeError:
                if recompute_lost_points:
                    init_points = cv2.goodFeaturesToTrack(old_gray,
                                                          mask=None,
                                                          **self.feature_params)
                    new_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                                   frame_gray,
                                                                   init_points,
                                                                   None,
                                                                   **self.lk_params)

                    good_points_only = new_points[st == 1]

                else:
                    break

            good_points_only = np.array(good_points_only)
            points.append(good_points_only)

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            init_points = good_points_only.reshape(-1, 1, 2)

        return np.array(points)
