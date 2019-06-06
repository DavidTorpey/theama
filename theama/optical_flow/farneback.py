"""
Author: Ziyad Jappie

License: Apache 2.0

Module containing a Python implementation of the Farneback
algorithm to compute optical flow.
"""

import numpy as np
import cv2


class Farneback(object):
    """
    Class for the implementation of the
    Farneback optical flow algorithm.
    """

    def perform_optical_flow(self, video):
        """Function to compute optical flow by using
        the Farneback algorithm. This is an example of dense optical flow
        which computes the optical flow for all points in a given frame.

        Note: the size of the resultant array is one less than the size of
        the original video since optical flow is computed between two
        successive frames.

        Args:
            video: is the video fed in as a numpy array.
        """
        if not isinstance(video, np.ndarray):
            raise Exception("Not a numpy array")
        if len(video.shape) < 3 or len(video.shape) > 4:
            raise Exception("Not a video numpy file")

        video = video.astype('uint8')
        frames, _, _, _ = video.shape
        previous_frame = cv2.cvtColor(video[0], cv2.COLOR_BGR2GRAY)
        flows = []
        for i in range(1, frames):
            current_frame = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(flow)
            previous_frame = current_frame

        return np.array(flows)
